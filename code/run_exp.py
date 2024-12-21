import subprocess

def main():
        
    scripts = [
        "run_train.py", 
        "run_inference.py", 
        "eval/eval/eval.py"
    ]

    for script in scripts:
        subprocess.run(["python", script], check=True)

if __name__ == '__main__':
    main()
