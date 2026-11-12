import argparse
import os
import queue
import random
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pygame


ROOT = Path(__file__).resolve().parent
GRID = 16
CELL = 30
BOARD_SIZE = GRID * CELL
OBSERVATION_SIZE = GRID * GRID + 10
LEFT, RIGHT, UP, DOWN = range(4)
VECTORS = {LEFT: (-1, 0), RIGHT: (1, 0), UP: (0, 1), DOWN: (0, -1)}
OPPOSITE = {LEFT: RIGHT, RIGHT: LEFT, UP: DOWN, DOWN: UP}

BG = (16, 25, 23)
PANEL = (28, 40, 36)
PANEL_ALT = (37, 52, 47)
GRID_LINE = (48, 65, 58)
TEXT = (242, 245, 236)
MUTED = (154, 169, 159)
LIME = (197, 242, 103)
CORAL = (255, 111, 91)
CYAN = (119, 205, 211)
BOARD_X = 42
BOARD_Y = 166
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 720


class SnakeWorld:
	def __init__(self):
		self.best = 0
		self.reset()

	def reset(self):
		center = GRID // 2
		self.snake = [(center, center), (center - 1, center), (center - 2, center)]
		self.direction = RIGHT
		self.score = 0
		self.steps = 0
		self.game_over = False
		self.won = False
		self.food = self._new_food()

	def _new_food(self):
		available = [(x, y) for y in range(GRID) for x in range(GRID) if (x, y) not in self.snake]
		return random.choice(available) if available else None

	def step(self, action):
		if self.game_over:
			return False
		if action == OPPOSITE[self.direction]:
			action = self.direction

		dx, dy = VECTORS[action]
		head = (self.snake[0][0] + dx, self.snake[0][1] + dy)
		eating = head == self.food
		collision_body = self.snake if eating else self.snake[:-1]
		if not (0 <= head[0] < GRID and 0 <= head[1] < GRID) or head in collision_body:
			self.game_over = True
			return False

		self.direction = action
		self.snake.insert(0, head)
		if eating:
			self.score += 1
			self.best = max(self.best, self.score)
			self.food = self._new_food()
			if self.food is None:
				self.game_over = True
				self.won = True
		else:
			self.snake.pop()
		self.steps += 1
		return True

	def observation(self):
		values = [0.0] * OBSERVATION_SIZE
		head_x, head_y = self.snake[0]
		for x, y in self.snake:
			values[y * GRID + x] = 1.0
		values[GRID * GRID] = head_x / (GRID - 1)
		values[GRID * GRID + 1] = head_y / (GRID - 1)
		values[GRID * GRID + 2] = self.food[0] / (GRID - 1)
		values[GRID * GRID + 3] = self.food[1] / (GRID - 1)
		values[GRID * GRID + 4 + self.direction] = 1.0
		values[GRID * GRID + 8] = self.snake[-1][0] / (GRID - 1)
		values[GRID * GRID + 9] = self.snake[-1][1] / (GRID - 1)
		return values


class AgentBridge:
	SOURCES = [
		"Snake.c",
		"arena.c",
		"random.c",
		"environment.c",
		"matrix.c",
		"model.c",
		"training.c",
	]
	HEADERS = ["arena.h", "common.h", "environment.h", "matrix.h", "model.h", "random.h", "training.h"]

	def __init__(self, epochs, retrain=False):
		self.epochs = epochs
		self.retrain = retrain
		self.events = queue.Queue()
		self.process = None
		self.thread = None

	def start(self):
		self.thread = threading.Thread(target=self._run, daemon=True)
		self.thread.start()

	def _run(self):
		executable = ROOT / ("Snake.exe" if os.name == "nt" else "Snake")
		if os.environ.get("SNAKE_PREBUILT") == "1":
			if not executable.is_file():
				self.events.put(("error", "Prebuilt C agent was not found in the application image."))
				return
		else:
			build_inputs = [ROOT / path for path in self.SOURCES + self.HEADERS]
			if not executable.exists() or any(path.stat().st_mtime > executable.stat().st_mtime for path in build_inputs):
				compiler = shutil.which("gcc")
				if compiler is None:
					self.events.put(("error", "gcc was not found on PATH."))
					return
				command = [compiler, "-std=c11", "-Wall", "-Wextra", "-O3", *self.SOURCES, "-lm", "-o", str(executable)]
				flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
				result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, creationflags=flags)
				if result.returncode != 0:
					message = result.stderr.strip() or result.stdout.strip() or "C build failed."
					self.events.put(("error", message[-500:]))
					return

		flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
		try:
			command = [str(executable), "--agent", str(self.epochs)]
			if self.retrain:
				command.append("--retrain")
			self.process = subprocess.Popen(
				command,
				cwd=ROOT,
				stdin=subprocess.PIPE,
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
				bufsize=1,
				creationflags=flags,
			)
			ready = False
			for line in self.process.stdout:
				line = line.strip()
				if line == "READY":
					ready = True
					self.events.put(("ready", None))
					break
				if line.startswith("Epoch "):
					try:
						epoch = int(line.split("|", 1)[0].split()[1]) + 1
						if epoch == 1 or epoch % 10 == 0 or epoch == self.epochs:
							self.events.put(("progress", (epoch, self.epochs)))
					except (IndexError, ValueError):
						pass

			if not ready:
				self.events.put(("error", "The C agent exited before becoming ready."))
				return

			for line in self.process.stdout:
				line = line.strip()
				if line.startswith("ACTION "):
					try:
						self.events.put(("action", int(line.split()[1])))
					except (IndexError, ValueError):
						self.events.put(("error", "The C agent returned an invalid action."))
				elif line.startswith("ERROR "):
					self.events.put(("error", line[6:]))
		except OSError as error:
			self.events.put(("error", str(error)))

	def send(self, observation):
		if self.process is None or self.process.poll() is not None:
			return False
		try:
			self.process.stdin.write(" ".join(f"{value:.8g}" for value in observation) + "\n")
			self.process.stdin.flush()
			return True
		except (BrokenPipeError, OSError):
			return False

	def stop(self):
		if self.process is not None and self.process.poll() is None:
			self.process.terminate()
			try:
				self.process.wait(timeout=2)
			except subprocess.TimeoutExpired:
				self.process.kill()


def font(size, bold=False):
	return pygame.font.SysFont("bahnschrift", size, bold=bold)


def draw_text(surface, typeface, text, position, color=TEXT):
	surface.blit(typeface.render(text, True, color), position)


def draw_button(surface, rect, label, active, typeface):
	fill = LIME if active else PANEL_ALT
	ink = BG if active else TEXT
	pygame.draw.rect(surface, fill, rect, border_radius=8)
	label_surface = typeface.render(label, True, ink)
	surface.blit(label_surface, label_surface.get_rect(center=rect.center))


def draw_game(screen, world, mode, paused, bridge_state, progress, error, speed):
	screen.fill(BG)
	small = font(13, True)
	label = font(15, True)
	body = font(18)
	heading = font(44, True)
	number = font(34, True)

	draw_text(screen, small, "REINFORCE  /  FIELD 01", (44, 30), CYAN)
	draw_text(screen, heading, "SNAKE", (42, 52))
	draw_text(screen, body, "A C-trained policy, live on the grid.", (224, 76), MUTED)
	pygame.draw.line(screen, GRID_LINE, (42, 126), (1058, 126), 1)

	board_rect = pygame.Rect(BOARD_X, BOARD_Y, BOARD_SIZE, BOARD_SIZE)
	pygame.draw.rect(screen, PANEL, board_rect, border_radius=10)
	for cell_y in range(GRID):
		for cell_x in range(GRID):
			x = BOARD_X + cell_x * CELL
			y = BOARD_Y + (GRID - 1 - cell_y) * CELL
			tile = pygame.Rect(x + 2, y + 2, CELL - 4, CELL - 4)
			pygame.draw.rect(screen, (31, 45, 40), tile, border_radius=5)

	if world.food is not None:
		food_x, food_y = world.food
		center = (BOARD_X + food_x * CELL + CELL // 2, BOARD_Y + (GRID - 1 - food_y) * CELL + CELL // 2)
		pygame.draw.circle(screen, CORAL, center, 10)
		pygame.draw.circle(screen, (255, 183, 142), (center[0] - 3, center[1] - 4), 3)

	for segment_index, (x, y) in enumerate(world.snake):
		rect = pygame.Rect(BOARD_X + x * CELL + 5, BOARD_Y + (GRID - 1 - y) * CELL + 5, CELL - 10, CELL - 10)
		color = (226, 255, 161) if segment_index == 0 else LIME
		pygame.draw.rect(screen, color, rect, border_radius=8)
		if segment_index == 0:
			dx, dy = VECTORS[world.direction]
			center_x, center_y = rect.center
			eye_x = center_x + dx * 10
			eye_y = center_y - dy * 10
			pygame.draw.circle(screen, BG, (eye_x - dy * 6, eye_y - dx * 6), 3)
			pygame.draw.circle(screen, BG, (eye_x + dy * 6, eye_y + dx * 6), 3)

	if world.game_over:
		overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
		overlay.fill((12, 19, 17, 178))
		screen.blit(overlay, board_rect.topleft)
		message = "GRID CLEARED" if world.won else "RUN ENDED"
		message_surface = font(31, True).render(message, True, LIME if world.won else CORAL)
		screen.blit(message_surface, message_surface.get_rect(center=(board_rect.centerx, board_rect.centery - 20)))
		restart_surface = body.render("Press R to run it back", True, TEXT)
		screen.blit(restart_surface, restart_surface.get_rect(center=(board_rect.centerx, board_rect.centery + 24)))
	elif paused:
		overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
		overlay.fill((12, 19, 17, 178))
		screen.blit(overlay, board_rect.topleft)
		draw_text(screen, font(32, True), "PAUSED", (BOARD_X + 148, BOARD_Y + 182))

	panel_x = 580
	panel_width = WINDOW_WIDTH - panel_x - 42
	pygame.draw.rect(screen, PANEL, (panel_x, BOARD_Y, panel_width, 150), border_radius=10)
	draw_text(screen, small, "PLAY MODE", (panel_x + 20, BOARD_Y + 18), MUTED)
	human_rect = pygame.Rect(panel_x + 20, BOARD_Y + 48, 150, 42)
	ai_rect = pygame.Rect(panel_x + 180, BOARD_Y + 48, 190, 42)
	draw_button(screen, human_rect, "HUMAN", mode == "human", label)
	draw_button(screen, ai_rect, "C POLICY", mode == "ai", label)
	status_y = BOARD_Y + 106
	if error:
		draw_text(screen, small, "AGENT ERROR", (panel_x + 20, status_y), CORAL)
	elif bridge_state == "ready":
		draw_text(screen, small, "POLICY ONLINE", (panel_x + 20, status_y), CYAN)
	elif bridge_state == "training":
		percentage = int(100 * progress[0] / max(progress[1], 1))
		draw_text(screen, small, f"TRAINING C POLICY  {percentage}%", (panel_x + 20, status_y), CYAN)
		bar = pygame.Rect(panel_x + 248, status_y + 2, 150, 9)
		pygame.draw.rect(screen, PANEL_ALT, bar, border_radius=4)
		pygame.draw.rect(screen, CYAN, (bar.x, bar.y, int(bar.width * percentage / 100), bar.height), border_radius=4)
	else:
		status = "READY WHEN YOU ARE" if mode == "human" else "STARTING C AGENT"
		draw_text(screen, small, status, (panel_x + 20, status_y), MUTED)
	if error:
		draw_text(screen, font(12), error[:62], (panel_x + 20, status_y + 20), MUTED)

	pygame.draw.rect(screen, PANEL, (panel_x, BOARD_Y + 170, panel_width, 144), border_radius=10)
	draw_text(screen, small, "RUN STATS", (panel_x + 20, BOARD_Y + 190), MUTED)
	draw_text(screen, small, "FOOD", (panel_x + 20, BOARD_Y + 222), CYAN)
	draw_text(screen, number, f"{world.score:02d}", (panel_x + 20, BOARD_Y + 242))
	draw_text(screen, small, "LENGTH", (panel_x + 165, BOARD_Y + 222), MUTED)
	draw_text(screen, number, f"{len(world.snake):02d}", (panel_x + 165, BOARD_Y + 242))
	draw_text(screen, small, "BEST", (panel_x + 310, BOARD_Y + 222), CORAL)
	draw_text(screen, number, f"{world.best:02d}", (panel_x + 310, BOARD_Y + 242))

	pygame.draw.rect(screen, PANEL, (panel_x, BOARD_Y + 334, panel_width, 122), border_radius=10)
	draw_text(screen, small, "CONTROLS", (panel_x + 20, BOARD_Y + 352), MUTED)
	draw_text(screen, body, "ARROWS / WASD  Move", (panel_x + 20, BOARD_Y + 380), TEXT)
	draw_text(screen, body, "SPACE  Pause     R  Restart", (panel_x + 20, BOARD_Y + 408), TEXT)
	draw_text(screen, small, f"STEP RATE  {speed:.1f} / SEC", (panel_x + 20, BOARD_Y + 478), CYAN)
	draw_text(screen, small, "[ - ] SLOWER        [ + ] FASTER", (panel_x + 20, BOARD_Y + 502), MUTED)

	pygame.display.flip()
	return human_rect, ai_rect


def main():
	parser = argparse.ArgumentParser(description="Play Snake yourself or watch the C REINFORCE policy.")
	parser.add_argument("--epochs", type=int, default=25, help="C policy training epochs before AI play (default: 25).")
	parser.add_argument("--ai", action="store_true", help="Start in C-policy mode and train immediately.")
	parser.add_argument("--retrain", action="store_true", help="Ignore the saved policy and train it again.")
	args = parser.parse_args()
	if args.epochs < 0 or args.epochs > 1000000:
		parser.error("--epochs must be between 0 and 1000000")

	pygame.init()
	pygame.display.set_caption("Snake / REINFORCE")
	screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
	clock = pygame.time.Clock()

	world = SnakeWorld()
	mode = "human"
	bridge = None
	bridge_state = "idle"
	progress = (0, args.epochs)
	error = ""
	paused = False
	awaiting_action = False
	discard_action = False
	speed = 45.0
	next_step = time.monotonic() + 1.0 / speed
	human_button = pygame.Rect(0, 0, 0, 0)
	ai_button = pygame.Rect(0, 0, 0, 0)
	running = True

	def choose_mode(new_mode):
		nonlocal mode, bridge, bridge_state, error, awaiting_action, discard_action
		if new_mode == "ai" and bridge is None:
			bridge = AgentBridge(args.epochs, args.retrain)
			bridge_state = "training"
			bridge.start()
		mode = new_mode
		error = ""
		if awaiting_action:
			discard_action = True

	if args.ai:
		choose_mode("ai")

	while running:
		now = time.monotonic()
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
				if human_button.collidepoint(event.pos):
					choose_mode("human")
				elif ai_button.collidepoint(event.pos):
					choose_mode("ai")
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					running = False
				elif event.key == pygame.K_SPACE:
					paused = not paused
					next_step = now + 1.0 / speed
				elif event.key == pygame.K_r:
					world.reset()
					awaiting_action = awaiting_action or discard_action
					discard_action = awaiting_action
					next_step = now + 1.0 / speed
				elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
					speed = max(2.0, speed - 1.0)
					next_step = now + 1.0 / speed
				elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
					speed = min(45.0, speed + 1.0)
					next_step = now + 1.0 / speed
				elif mode == "human":
					key_actions = {
						pygame.K_LEFT: LEFT,
						pygame.K_a: LEFT,
						pygame.K_RIGHT: RIGHT,
						pygame.K_d: RIGHT,
						pygame.K_UP: UP,
						pygame.K_w: UP,
						pygame.K_DOWN: DOWN,
						pygame.K_s: DOWN,
					}
					action = key_actions.get(event.key)
					if action is not None and action != OPPOSITE[world.direction]:
						world.direction = action

		if bridge is not None:
			while True:
				try:
					kind, data = bridge.events.get_nowait()
				except queue.Empty:
					break
				if kind == "ready":
					bridge_state = "ready"
					next_step = now
				elif kind == "progress":
					progress = data
				elif kind == "action":
					awaiting_action = False
					if discard_action:
						discard_action = False
					elif mode == "ai" and not paused and not world.game_over:
						world.step(data)
					next_step = now + 1.0 / speed
				elif kind == "error":
					error = data
					bridge_state = "error"
					awaiting_action = False

		if not paused and not world.game_over:
			if mode == "human" and now >= next_step:
				world.step(world.direction)
				next_step = now + 1.0 / speed
			elif mode == "ai" and bridge_state == "ready" and not awaiting_action and now >= next_step:
				awaiting_action = bridge.send(world.observation())
				if not awaiting_action:
					error = "Could not send state to the C agent."
					bridge_state = "error"

		human_button, ai_button = draw_game(screen, world, mode, paused, bridge_state, progress, error, speed)
		clock.tick(60)

	if bridge is not None:
		bridge.stop()
	pygame.quit()


if __name__ == "__main__":
	main()
