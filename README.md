# 2R_Plana_Robot_Motion_Planning_using-PRM
This project implements an interactive visualization system for planar 2R robotic arm motion planning. The system allows users to select target positions through mouse clicks, and automatically generates collision-free trajectories using the Probabilistic Roadmap Method (PRM) algorithm. 

Animation result:
![result](https://github.com/user-attachments/assets/af15eac7-0a7b-4661-aa5c-bab0b79f0370)

## Setting Up Execute Environment

Suggest to use Anaconda/miniconda.

```bash
# if use the env in conda(optional)
conda create -n mp python=3.9
conda activate mp

# install the program's dependency
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

After you finish the above set step, execute

```bash
python main.py
```

Then the program will run. Click on the right for automatic path planning

Search results:
![result2](https://github.com/user-attachments/assets/87757db9-9b77-4bc0-8a89-08d521199b78)


