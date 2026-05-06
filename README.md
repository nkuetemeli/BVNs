# Bernstein-Vazirani Networks: Quantum Machine Learning by Interference
This code can be used to reproduce the experiments and results of our paper <br/>
**Bernstein-Vazirani Networks: Quantum Machine Learning by Interference**.

We introduce Bernstein–Vazirani Networks (BVNs), a non‑variational quantum machine learning framework that leverages quantum interference for supervised learning, demonstrated on vision and representation tasks.
In their standard form, BVNs follow the principle of quantum Fourier sampling: 
labelled data are placed in superposition and interfered in the Fourier basis to extract globally informative features. 
We then define generalised BVNs enabling interference in problem-adapted bases and yielding more expressive models under the same measurement budget as the standard setting. 
BVNs achieve universal function approximation through complete interference bases, while training of BVNs is gradient-free. 
Experiments on synthetic and real‑world classification tasks, as well as neural implicit image representations, show strong generalisation and competitive performance compared to classical and quantum baselines. 

# Install
The code depends on the Python packages 
[numpy](https://numpy.org/install/), 
[torch](https://pytorch.org/),
[pennylane](https://pennylane.ai/install),
and (for plots)
[matplotlib](https://pypi.org/project/matplotlib/).

- Please download the repository and install the requirements in `requirements.txt` or refer to the product pages for reference.

- Once you satisfied the dependency, run `python -m pip install .` inside the directory.

Move to the `src` folder to run the subsequent commands.

# Example

    # Run BVNs on some test learning cases
    python bvn_relu_act.py         # BVN with Reversible RuLU-MLP as expressive representation
    python bvn_rect_act.py         # BVN with Rectangles as expressive representation
    python bvn_img_fitting.py      # BVN with Rectangles as expressive representation on image representation

# Citation
If you find this work useful, please cite the article [Article URL](#).
