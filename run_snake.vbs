Set fileSystem = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

root = fileSystem.GetParentFolderName(WScript.ScriptFullName)
pythonw = root & "\.venv\Scripts\pythonw.exe"

If Not fileSystem.FileExists(pythonw) Then
    MsgBox "Project virtual environment not found. Create .venv and install pygame first.", vbExclamation, "Snake"
    WScript.Quit 1
End If

shell.CurrentDirectory = root
quote = Chr(34)
command = quote & pythonw & quote & " " & quote & root & "\snake.py" & quote & " --ai"
shell.Run command, 0, False