from datetime import datetime
from pathlib import Path
from src.origin.gspan import gSpan

def main():
    project_path = Path(__file__).parent.resolve()
    data_path = project_path / "graphdata" / "graph.data"
    gs = gSpan(data_path, min_support=100, min_num_vertices=2)
    start = datetime.now().timestamp()
    gs.run()
    print(datetime.now().timestamp() - start)

# 5000  27.08 s     51.48 s
# 3000  34.38 s  	69.07 s
# 1000  1 m 48 s    3 m 49 s
# 600   3 m 14 s    7 m 29 s
# 400   5 m 26 s    12 m 53 s
# 100   1 h 3 m

if __name__ == "__main__":
    main()
