# Three_body_problem
The Concept
This project explores the Three-Body Problem—a classic physics puzzle where three celestial bodies move under the influence of mutual gravitational attraction. Because the system is chaotic, it is analytically unsolvable.In this project, I bridge Classical Mechanics with Machine Learning to determine if a system’s initial state will lead to a stable orbit or chaotic escape.

🧠 The Scientific LogicNumerical Integration: 
The simulator uses a step-based approach to solve gravitational force vectors.

Chaos Theory: 
The project visualizes "Sensitivity to Initial Conditions," where a change as small as $0.001$ units in the starting position leads to exponentially different outcomes.

Physics-Informed ML: A MLPClassifier (Neural Network) is trained to classify orbital stability based on raw vector states $[x, y, v_x, v_y]$, effectively learning the "rules" of the gravitational dance.
🛠️ Tech StackLanguage: Python 3.13Computation: numpy (Vectorized gravitational math)
AI/ML: scikit-learn (Multi-Layer Perceptron for classification)
Visualization: matplotlib (Animations and chaos sensitivity plots)
📊 Key VisualizationsTrajectory Map: Visualizing the complex "bird's nest" of paths in the chaotic system.
Sensitivity Plot: A log-scale chart demonstrating how quickly identical systems diverge when perturbed by a tiny fraction.🚀 Getting StartedClone the repo: git clone https://github.com/yourusername/yourrepo.
gitSetup environment: python -m venv venv && source venv/bin/activateInstall deps: pip install numpy pandas scikit-learn matplotlibGenerate data: python generate_data.py (Creates the dataset)
Train AI: python train_ai.py (Outputs prediction accuracy)
🔮 Future ImprovementsPINNs (Physics-Informed Neural Networks): Integrating the laws of conservation of energy directly into the AI's loss function.3D Visualization: Moving the simulation from a 2D plane to a 3D coordinate space.💡 Why this project?This project demonstrates the ability to generate synthetic datasets, handle high-precision numerical computations, and apply machine learning to non-linear physical systems.
