# ts-torch

a fork of [webgpu-torch](https://github.com/praeclarum/webgpu-torch) by [praeclarum](https://twitter.com/praeclarum)

---

Goal is to develop an image diffuser based on [this implementation](https://github.com/dome272/Diffusion-Models-pytorch)
For this task we expanded webgpu-torch to include a variety of functions relating to distributions and attention

---

## Get Started

Install all package dependencies 

```
npm install
```

Then run the following to start a dev build

```
npm run serve
```

At which point you will be prompted for a test file. These can be generated using `@/scripts/gen.py`, test generators are defined in `@/scripts/tests,py`
Tests are handled in `@/scripts/tester.ts`. If you're contributing, this is where you can create test handlers for any functions you build.
To see it in action, ff to write some tests & handlers. ping me if you have any Qs

## Directory Structure

All the torch files are held in `/torch`. We try to match pytorch's API as best as ts allows. In here are a couple notable groups:
- `tensor.ts`: Where the Tensor object is defined
- `nn_*`: modules, such as Linear and MultiheadAttention (WIP)
- `op(s)_*`: operations definitions, you'll find a lot of the custom code in `ops_artisanal`
- `kernels_*`: GPU operation (kernel) definitions, this where functions are typically implemented

## Contribution Notes
- ALWAYS BRANCH before writing new code, then make a PR to get it put into master
- ff to ping me anytime
- [pytorch ource code](https://github.com/pytorch/pytorch/tree/main), for reference
- [webGPU shading language](https://www.w3.org/TR/WGSL/), for reference