# Blockblast

### setup
If the modules **blockblast** or **qdn** create an import error :
```bash
uv pip install -e .
```
### project architecture
```
block-blast-rl/
├── src/                        
│   ├── blockblast/             
│   │   ├── __init__.py         
│   │   ├── block_blast_env.py    # Single-piece environment version
│   │   └── block_blast_3p_env.py  # Three-piece environment version
│   └── dqn/                    
│       ├── __init__.py         
│       ├── agent.py              # DQN Logic (Replay Buffer, Target Network)
│       └── model.py              # CNN Architecture
├── notebooks/                  
│   ├── training.ipynb            # Agent training loops
│   └── evaluation.ipynb          # Performance testing and video rendering
├── checkpoints/                
├── videos/                     
├── .venv/                      
├── pyproject.toml              
└── .gitignore
```
