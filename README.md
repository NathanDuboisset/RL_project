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
│   ├── notebook.ipynb            
│   │   ...
│   └── notebook.ipynb         
├── checkpoints/                
├── .venv/                      
├── pyproject.toml              
└── .gitignore
```
