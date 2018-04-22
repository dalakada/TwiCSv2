from pathlib import Path

asm_paths = []
for pth in Path.cwd().iterdir():
	with open(pth) as f:
	    lines = f.readlines()
        f=open(pth,'r')
        content = f.readlines()


print(content)

with open('filename') as f:
    lines = f.readlines()