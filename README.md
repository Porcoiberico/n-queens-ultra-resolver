# N-Queens Ultra-Efficient Solver

A Python algorithm that resolves the N-Queens problem. It applies **AC-3 preprocessing**, an **extreme memetic algorithm**, and various heuristics like **MRV** (Minimum Remaining Values) and more.

It resolves a problem of **10,000 initial queens with 4,000 static queens** in a very short time.

## Key Techniques
- **AC-3 Preprocessing:** Arc consistency constraint propagation to prune invalid domains prior to search.
- **Extreme Memetic Algorithm:** Hybrid evolutionary optimization combined with intensive local search.
- **Heuristics:** Minimum Remaining Values (MRV) and domain-specific ordering heuristics.

## Usage
Run the main solver with Python 3:

`python resolver.py`

## Author & License
Created by Miguel García Alcover.  
This project is licensed under the [MIT License](LICENSE).
