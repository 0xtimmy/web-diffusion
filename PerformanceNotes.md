# Function Specific

### Repeat

*original strategy:* call cat multiple times in order to acheive repeat
*new strategy:* dedicated GPU kernel

tested w/ input shape `[2, 8, 16, 32]` and repeat shape `[16, 8, 4, 2]`

**result:** 20x faster