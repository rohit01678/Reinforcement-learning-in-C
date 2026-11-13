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
		self.last_reward = 0.0
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
		old_distance = abs(self.food[0] - self.snake[0][0]) + abs(self.food[1] - self.snake[0][1])
		new_distance = abs(self.food[0] - head[0]) + abs(self.food[1] - head[1])
		reward = 0.1 * (old_distance - new_distance) - 0.01
		eating = head == self.food
		collision_body = self.snake if eating else self.snake[:-1]
		if not (0 <= head[0] < GRID and 0 <= head[1] < GRID) or head in collision_body:
			self.game_over = True
			self.last_reward = reward - 10.0
			return False

		self.direction = action
		self.snake.insert(0, head)
		if eating:
			reward += 20.0
			self.score += 1
			self.best = max(self.best, self.score)
			self.food = self._new_food()
			if self.food is None:
				self.game_over = True
				self.won = True
		else:
			self.snake.pop()
		self.steps += 1
		self.last_reward = reward
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
		self.has_pending_action = False

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
				command = [compiler, "-std=c11", "-Wall", "-Wextra", "-O3", "-march=native", "-funroll-loops", "-ffast-math", *self.SOURCES, "-lm", "-o", str(executable)]
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
				elif line.startswith("LEARNED "):
					try:
						self.events.put(("learned", int(line.split()[1])))
					except (IndexError, ValueError):
						self.events.put(("error", "The C agent returned an invalid learning update."))
				elif line.startswith("ERROR "):
					self.events.put(("error", line[6:]))
		except OSError as error:
			self.events.put(("error", str(error)))

	def send(self, observation):
		if self.process is None or self.process.poll() is not None:
			return False
		try:
			self.process.stdin.write("STEP " + " ".join(f"{value:.8g}" for value in observation) + "\n")
			self.process.stdin.flush()
			self.has_pending_action = True
			return True
		except (BrokenPipeError, OSError):
			return False

	def feedback(self, reward, done):
		if self.process is None or self.process.poll() is not None or not self.has_pending_action:
			return False
		try:
			self.process.stdin.write(f"FEEDBACK {reward:.8g} {int(done)}\n")
			self.process.stdin.flush()
			self.has_pending_action = False
			return True
		except (BrokenPipeError, OSError):
			return False

	def end_episode(self):
		if self.process is None or self.process.poll() is not None or self.has_pending_action:
			return False
		try:
			self.process.stdin.write("END\n")
			self.process.stdin.flush()
			return True
		except (BrokenPipeError, OSError):
			return False

	def abort_episode(self):
		if self.process is None or self.process.poll() is not None:
			return False
		try:
			self.process.stdin.write("ABORT\n")
			self.process.stdin.flush()
			self.has_pending_action = False
			return True
		except (BrokenPipeError, OSError):
			return False

	def stop(self):
		if self.process is None or self.process.poll() is not None:
			return
		if self.has_pending_action:
			self.abort_episode()
		else:
			self.end_episode()
		if self.process.stdin is not None:
			self.process.stdin.close()
		try:
			self.process.wait(timeout=5)
		except subprocess.TimeoutExpired:
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


def draw_game(screen, world, mode, paused, bridge_state, progress, error, speed, speed_text, speed_input_active, speed_input_error, policy_updates):
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
	human_rect = pygame.Rect(panel_x + 20, BOARD_Y + 48, 120, 42)
	ai_rect = pygame.Rect(panel_x + 148, BOARD_Y + 48, 130, 42)
	retrain_rect = pygame.Rect(panel_x + 286, BOARD_Y + 48, 170, 42)
	draw_button(screen, human_rect, "HUMAN", mode == "human", label)
	draw_button(screen, ai_rect, "C POLICY", mode == "ai", label)
	draw_button(screen, retrain_rect, "AUTO TRAIN", bridge_state == "training", label)
	status_y = BOARD_Y + 106
	if error:
		draw_text(screen, small, "AGENT ERROR", (panel_x + 20, status_y), CORAL)
	elif bridge_state == "ready":
		status = f"POLICY UPDATED x{policy_updates}" if policy_updates else "POLICY ONLINE"
		draw_text(screen, small, status, (panel_x + 20, status_y), CYAN)
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
	draw_text(screen, small, "STEPS / SEC", (panel_x + 20, BOARD_Y + 478), CYAN)
	speed_input_rect = pygame.Rect(panel_x + 150, BOARD_Y + 466, 76, 30)
	pygame.draw.rect(screen, PANEL_ALT, speed_input_rect, border_radius=6)
	pygame.draw.rect(screen, CYAN if speed_input_active else GRID_LINE, speed_input_rect, 2, border_radius=6)
	input_text = speed_text if speed_input_active else str(int(speed))
	if speed_input_active:
		input_text += "|"
	input_surface = small.render(input_text, True, TEXT)
	screen.blit(input_surface, input_surface.get_rect(center=speed_input_rect.center))
	if speed_input_error:
		speed_hint = "Enter a value from 2 to 45"
	elif speed_input_active:
		speed_hint = "Type 2-45; Enter applies"
	else:
		speed_hint = "Click to set; [ - ] / [ + ] adjust"
	draw_text(screen, small, speed_hint, (panel_x + 20, BOARD_Y + 502), CORAL if speed_input_error else MUTED)

	pygame.display.flip()
	return human_rect, ai_rect, retrain_rect, speed_input_rect


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
	speed_text = str(int(speed))
	speed_input_active = False
	speed_input_error = False
	policy_updates = 0
	next_step = time.monotonic() + 1.0 / speed
	human_button = pygame.Rect(0, 0, 0, 0)
	ai_button = pygame.Rect(0, 0, 0, 0)
	retrain_button = pygame.Rect(0, 0, 0, 0)
	speed_input_rect = pygame.Rect(0, 0, 0, 0)
	running = True

	def choose_mode(new_mode, force_retrain=False):
		nonlocal mode, bridge, bridge_state, error, awaiting_action, discard_action, progress, policy_updates
		error = ""
		if new_mode == "ai" and force_retrain:
			if bridge is not None:
				bridge.stop()
				bridge = None
			bridge_state = "idle"
			awaiting_action = False
			discard_action = False
			progress = (0, args.epochs)
			policy_updates = 0
		if new_mode == "ai" and bridge is None:
			bridge = AgentBridge(args.epochs, force_retrain or args.retrain)
			bridge_state = "training"
			bridge.start()
		if new_mode != mode and mode == "ai" and bridge is not None and bridge_state == "ready":
			if awaiting_action:
				if not bridge.abort_episode():
					error = "Could not discard the pending C policy action."
				discard_action = True
			elif not bridge.end_episode():
				error = "Could not send episode feedback to the C agent."
		mode = new_mode

	if args.ai:
		choose_mode("ai")

	while running:
		now = time.monotonic()
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
				if speed_input_rect.collidepoint(event.pos):
					speed_input_active = True
					speed_text = ""
					speed_input_error = False
				else:
					speed_input_active = False
					speed_input_error = False
					speed_text = str(int(speed))
				if human_button.collidepoint(event.pos):
					choose_mode("human")
				elif ai_button.collidepoint(event.pos):
					choose_mode("ai")
				elif retrain_button.collidepoint(event.pos):
					choose_mode("ai", force_retrain=True)
			elif event.type == pygame.KEYDOWN:
				if speed_input_active:
					if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
						if speed_text.isdigit() and 2 <= int(speed_text) <= 45:
							speed = float(int(speed_text))
							next_step = now + 1.0 / speed
							speed_input_active = False
							speed_input_error = False
						else:
							speed_input_error = True
					elif event.key == pygame.K_ESCAPE:
						speed_input_active = False
						speed_input_error = False
						speed_text = str(int(speed))
					elif event.key == pygame.K_BACKSPACE:
						speed_text = speed_text[:-1]
						speed_input_error = False
					elif event.unicode in "0123456789" and len(speed_text) < 2:
						speed_text += event.unicode
						speed_input_error = False
					continue
				elif event.key == pygame.K_ESCAPE:
					running = False
				elif event.key == pygame.K_SPACE:
					paused = not paused
					next_step = now + 1.0 / speed
				elif event.key == pygame.K_r:
					if mode == "ai" and bridge is not None and bridge_state == "ready" and not world.game_over:
						if awaiting_action:
							if not bridge.abort_episode():
								error = "Could not discard the pending C policy action."
							discard_action = True
						elif not bridge.end_episode():
							error = "Could not send episode feedback to the C agent."
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
				elif kind == "learned":
					policy_updates += 1
				elif kind == "action":
					awaiting_action = False
					if discard_action:
						discard_action = False
					elif mode == "ai" and not paused and not world.game_over:
						world.step(data)
						if not bridge.feedback(world.last_reward, world.game_over):
							error = "Could not send episode feedback to the C agent."
							bridge_state = "error"
					else:
						if not bridge.abort_episode():
							error = "Could not discard the pending C policy action."
							bridge_state = "error"
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
				next_step = now + 1.0 / speed
				if not awaiting_action:
					error = "Could not send state to the C agent."
					bridge_state = "error"

		human_button, ai_button, retrain_button, speed_input_rect = draw_game(
			screen, world, mode, paused, bridge_state, progress, error,
			speed, speed_text, speed_input_active, speed_input_error, policy_updates,
		)
		clock.tick(60)

	if bridge is not None:
		bridge.stop()
	pygame.quit()


if __name__ == "__main__":
	main()
