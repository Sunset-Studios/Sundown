# Sundown Engine 🕹️

#### An extendible WebGPU game and simulation engine for fun, games and research.
![sundown_demo](https://github.com/user-attachments/assets/fb001dca-66dd-4ba5-b307-428cad52e441)

<img width="1583" height="885" alt="Screenshot 2025-11-18 132838" src="https://github.com/user-attachments/assets/9131c374-b200-49c0-940a-ad52afdacda4" />

<img width="1582" height="897" alt="Screenshot 2025-11-18 133035" src="https://github.com/user-attachments/assets/30e32ad7-1714-4aa1-8e38-667f0eee3aab" />

https://github.com/user-attachments/assets/3671857c-6eab-422b-81d8-8d6095874c5a

https://github.com/user-attachments/assets/2925ccb8-4484-48e6-a049-143ed0b5c944

Some of the current (code) features include:

* ⚡ WebGPU renderable abstractions
* ⚡ Flexible render graph for crafting render and compute pipelines
* ⚡ Simple, expressive, shader-based material system for crafting custom materials
* ⚡ Layered Gameplay simulation architecture for adding layered, modular functionality
* ⚡ Archetype-chunk ECS system for more efficient processing, using a fragment framework and TypedArrays
* ⚡ Simple, context-based input system, allowing you to set up different input schemes and contexts
* ⚡ Built-in PBR shaders
* ⚡ Entity-first instancing
* ⚡ Auto mesh instancing and draw batching of meshes using a specialized mesh task queue
* ⚡ Compute task queue for easily submitting compute shader work
* ⚡ MSDF text rendering
* ⚡ Configurable post-process stack
* ⚡ Immediate mode screen-space UI
* ⚡ Sparse Virtual Shadow Maps 
* ⚡ Dynamic TLAS BVH tree acceleration structure for physics, ray tracing and ray casting
* ⚡ Per-mesh triangle BLAS BVH for software ray tracing
* ⚡ Pseudo-bindless texture pools and material tables 
* ⚡ Software (Compute) path tracer with ReSTIR
* ⚡ 2-level radiance cache realtime GI using software path tracing 
* ⚡ Helpers for loading GTLFs, tracking performance scopes, named IDs, running frames and more.

Sundown also includes a simple but capable ML framework for running real-time AI experiments:
* ⚡ Simple gradient tape for backprop based learning
* ⚡ High-level, layer-based DAG subnet API for composing models from smaller subnetworks
* ⚡ Expanding library of activation functions, loss functions, optimizers and configurable layers
* ⚡ MasterMind class for orchestrating weight sharing, adaptation and real-time retraining of multiple models

### Installation


Make sure you have the latest version of [NodeJS](https://nodejs.org/en) installed. Clone this repository and make sure to `npm install` to get all the package dependencies.


```bash
> git clone git@github.com:Sunset-Studio/Sundown.git
> cd Sundown
> npm install
```

### Running


With the project cloned and all package dependencies installed, you're ready to run the project. There is an example **app.js** that is included from the top-level **index.html** file. Feel free to replace this with your own experiments or entry points.


To run the development project in a browser, use the npm `dev` command
```bash
> npm run dev
```


To run the development project in a Tauri instance, use the npm `devtop` command
```bash
> npm run devtop
```


### Packaging


You can package and distribute builds for the web or for desktop with the help of [Tauri](https://v2.tauri.app/).


To package for the web, just run the npm `build` command.
```bash
> npm run build
```


Then copy the resulting **index.html** file and **assets** and **engine** directories into your site's root.


To build executable Tauri packages, use the provided npm `make` command. This will create executable outputs in a top level *executables* directory.
```bash
> npm run make
```


### Contributing


Sundown is available for free under the MIT license. You can use and modify the engine for individual or commercial use (a reference or mention is still appreciated!) If you want to contribute features or fixes, please fork this repository and submit PRs. I am a one man team but will check any promising PRs as soon as I can. If you want to become a regular contributor feel free to DM me on [X](https://x.com/SunsetLearn) or shoot me an email at adrians.sanchez@sunsetlearn.com.
