Lets work on the carts benchmarks testing... Lets start by running the 
task-deps benchmarks with the small sizes. Lets make sure they work abd produce good results. We should have a verification logic on all the codes.

- /Users/randreshg/Documents/carts/external/carts-benchmarks/kastors/jacobi/jacobi-task-dep
- /Users/randreshg/Documents/carts/external/carts-benchmarks/kastors/jacobi/poisson-task/poisson-task.c
- /Users/randreshg/Documents/carts/external/carts-benchmarks/kastors/sparselu/sparselu-task
- /Users/randreshg/Documents/carts/external/carts-benchmarks/kastors/sparselu/sparselu-task-dep
- /Users/randreshg/Documents/carts/external/carts-benchmarks/kastors/strassen/strassen-task
- /Users/randreshg/Documents/carts/external/carts-benchmarks/kastors/strassen/strassen-task-dep

Things to consider if the code fails - Lets make sure
Remember that CARTS expectes only a single function se agressively need to inline the function. Lets make the function statics so we can inline them.

For now we dont support collapse, so simply remove that pragma for now... keep the codes as non collapsale for now...