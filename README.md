# Sundown Engine 🕹️


An extendible WebGPU game and simulation engine for fun, games and research. Some of the current (code) features include:

* ⚡ WebGPU renderable abstractions
* ⚡ Flexible render graph for crafting render and compute pipelines
* ⚡ Simple, expressive material system for crafting custom shaders and materials
* ⚡ Gameplay simulation layer system for adding layered, modular functionality
* ⚡ ECS system for more efficient processing, using TypedArrays where possible
* ⚡ Simple, context-based input system, allowing you to set up different input schemes and contexts
* ⚡ Built-in PBR shaders
* ⚡ Auto instancing and draw batching of meshes using a specialized mesh task queue
* ⚡ Compute task queue for easily submitting compute shader work
* ⚡ Helpers for loading GTLFs, tracking performance scopes, named IDs, running frames and more.

![Sundown Demo](./sundown_demo.gif)

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


To run the development project in an electron instance, use the npm `devtop` command
```bash
> npm run devtop
```


### Packaging


You can package and distribute builds for the web or for desktop with the help of [Electron Forge](https://www.electronforge.io/).


To package for the web, just run the npm `build` command.
```bash
> npm run build
```


Then copy the resulting **index.html** file and **assets** and **engine** directories into your site's root.


To build executable electron packages, use the provided npm `make` command. This will create executable outputs in a top level *executables* directory.
```bash
> npm run make
```


### Contributing


Sundown is available for free under the MIT license. You can use and modify the engine for individual or commercial use (a reference or mention is still appreciated!) If you want to contribute features or fixes, please fork this repository and submit PRs. I am a one man team but will check any promising PRs as soon as I can. If you want to become a regular contributor feel free to DM me on [X](https://x.com/SunsetLearn) or shoot me an email at adrians.sanchez@sunsetlearn.com.
