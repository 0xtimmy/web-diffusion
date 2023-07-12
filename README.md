# Web Diffusion - a bare-bones image diffsuer built with WebGPU.

Consumer hardware had long outgrown its demand to the point where, aside for gamers, there was really no reason to use technology beyond the intel i5. AI changed that: now consumers are running software that *requires* compute not normally met by personal computers. Chipmakers are already pushing out new products to make client side AI a possibility for developers, but for now you can make do with just a Ryzen 7 and some sort of GPU.

The launch of WebGPU gave hardware-accelerated applications access to the most powerful distribution tool in history: the internet. And any casual web surfer can run AI software locally, via google search.

WebDiffuser is typescript implementation of a bare bones image diffuser intended as an experiment in client side ML. It contains a small torch-like library of gpu-kernels and a very basic workload optimizer.

---

## Roadmap
- [X] WebGPU torch implementation
- [X] Diffusions Modules
- [X] [Pokemon Model](https://huggingface.co/datasets/huggan/pokemon)

- [ ] Performance improvements
- [ ] Support for Stable Diffusion Models

---

## Lampshading
- WebGPU is not yet standard across all browsers. This was tested in Google Chrome and Brave
- A GPU is required. This was tested on AMD graphics but should work on NVIDIA as well
- I spent my time coding and not training so the model itself is a little mid

## Get Started

Install all package dependencies 

```
npm install
```

Then run the following to start a dev build

```
npm run serve
```

Select the model you want to run; open dev tools (ctrl-shift-i) if you want to track progress.

---
## Acknoledgements

a fork of [webgpu-torch](https://github.com/praeclarum/webgpu-torch) by [praeclarum](https://twitter.com/praeclarum)
based on [this implementation](https://github.com/dome272/Diffusion-Models-pytorch)