V0. A kernel where each thread only compares and exchanges. This "eliminates" the 1:n 
innermost loop. Easy to write, but too many function calls and global synchronizations.

V1. Include the k inner loop in the kernel function. How do we handle the synchronization? 
Fewer calls, fewer global synchronizations. Faster than V0!

V2. Modify the kernel of V1 to work with local memory instead of global.
