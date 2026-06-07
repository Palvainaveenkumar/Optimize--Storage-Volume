import random
from SketchDataPlacement import CountMinSketch
all_movies = ['The Terminator (1984)', 'The Matrix (1999)', 'The Matrix Reloaded (2003)',
                  'The Matrix Revolutions (2003)', 'Tron (1982)', 'TRON: Legacy (2010)', 'Stargate (1994)',
                  'Coherence (2013)', 'The Chronicles of Riddick (2004)', 'Riddick (2013)']


sketch = CountMinSketch(8, 4)
for i in range(0,10):
    stream = all_movies[random.randint(0, random.randint(0, len(all_movies) - 1))]
    sketch.add(stream)
    print(stream)

for i in range(0,10):
    print(str(sketch.query(all_movies[i]))+" "+str(all_movies[i]))
print(sketch.len())
