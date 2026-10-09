# Model Schedule & Notes

### Schedule
- [ 
Attention visualization strategy: Layer-wise attention heatmaps (recommended)
Capture attention weights from each transformer layer
Map back to original table structure (32×238)
Show which temporal and feature regions the model focuses on
]

### Notes
- [

]

### Discussion
- [
    1. Which layers to visualize? : last layer
    2. Aggregation method?: mean across heads
    3. Visualization format?: matplotlib figures
    4. Should we visualize attention for both temporal and feature patches separately, or the concatenated version?: concatenated
]