We are going to be working on the carts-benchmarks repository.
/Users/randreshg/Documents/carts/external/carts-benchmarks

These correspond to thhe set of codes carts will run.
Lets analyze each example... Lets make sure each one of the run. 
Group for example name, for example:
- sw4lite folder should have the rhs4sg and vel4sg codes.

Each example should have a README.md file that describes the example, what it does, and how to run it.
Also make sure they have:
- a makefile that builds the code (with 'carts execute' command), make sure to add all flags and dependencies of the code
- then create 3 types of sizes for each example (this should be configured in the makefile)
  - small: 1000
  - medium: 10000
  - large: 100000


Check if we have duplicated codes....

then lets updae the readme.md of the project

Lets also create a carts-benchmarks sript, this will allows us to generate the binary for each example given an benchmark name, size, and and arts.cfg file we should generate a binary and auytomatically return the binary name, so we can later on run it