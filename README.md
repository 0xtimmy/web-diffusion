# Web DIffusion

a fork of [webgpu-torch](https://github.com/praeclarum/webgpu-torch) by [praeclarum](https://twitter.com/praeclarum)

---

A basic image diffuser based on [this implementation](https://github.com/dome272/Diffusion-Models-pytorch) built with WebGPU.

---

# Roadmap
- [X] WebGPU torch implementation
- [X] Diffusions Modules
- [X] [Pokemon Model](https://huggingface.co/datasets/huggan/pokemon)

- [ ] Performance improvements
- [ ] Support for Stable Diffusion Models

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

At which point you will be prompted for a test file. These can be generated using `@/scripts/gen.py`
Tests are handled in `@/scripts/tester.ts`. If you're contributing, this is where you can create test handlers for any functions you build.
To see it in action, ff to write some tests & handlers. ping me if you have any Qs

## Directory Structure

All the torch files are held in `/torch`. We try to match pytorch's API as best as ts allows. In here are a couple notable groups:
- `tensor.ts`: Where the Tensor object is defined
- `op(s)_*`: operations definitions, you'll find a lot of the custom code in `ops_artisanal`
- `kernels_*`: GPU operation (kernel) definitions, this where functions are typically implemented