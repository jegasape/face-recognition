from subprocess import run 

if __name__ == "__main__":
    [run(['python', s], capture_output=True, text=True) for s in ['video.py', 'trainer.py', 'recognition.py']]

