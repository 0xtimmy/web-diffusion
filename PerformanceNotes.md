# Model Optimimation

## `model.forward average times`
> improvement measured as `base_time / new_time`

| Date | description | duration | incremental improvement | total improvement
|----------|---------|--------- | ----------------------- | -----------------
| 07/12/23 | base | 36702ms | 1 | 1
| 07/17/23 | added repeat kernel | 9163ms | 4.00 | 4.00
| 07/18/23 | played with the workload optimzer equation | 8547ms | 1.07 | 4.29

# Function Specific

### `repeat`
> *original strategy:* call cat multiple times in order to acheive repeat\
> *new strategy:* dedicated GPU kernel

**test params:**
```
{
    input_shape: [2, 8, 16, 32],
    repeat_shape: [16, 8, 4, 2]
}
```

**result:** 20x faster