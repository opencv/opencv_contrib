## Introduction to "Plan" and "V4D"

### Overview of "Plan"
**Plan** is a computational graph engine built with C++20 templates, enabling developers to construct directed acyclic graphs (DAGs) from fragments of algorithms. By leveraging these graphs, Plan facilitates the optimization of parallel and concurrent algorithms, ensuring efficient resource utilization. The framework divides the lifetime of an algorithm into two distinct phases: **inference** and **execution**. 

- **Inference Phase:** During this phase, the computational graph is constructed by running the Plan implementation. This process organizes the algorithm's fragments and binds them to data, which may be classified as:
  - **Safe Data:** Member variables of the Plan.
  - **Shared Data:** External variables (e.g., global or static data).
  
  Functions and data are explicitly flagged as shared when necessary, adhering to Plan’s transparent approach to state management. The framework discourages hidden states, as they impede program integrity and graph optimization. 

- **Execution Phase:** This phase executes the constructed graph using the defined nodes and edges. Nodes typically represent algorithmic fragments such as functions or lambdas, while edges define data flow, supporting various access patterns (e.g., read, write, copy).

Plan also allows hierarchical composition, where one Plan may be composed of other sub-Plans. Special rules govern data sharing in such compositions to maintain performance and correctness. Currently, optimizations are limited to “best-effort” pipelining, with plans for more sophisticated enhancements.

### Overview of "V4D"
**V4D** is a versatile 2D/3D graphics runtime designed to integrate seamlessly with Plan. Built atop OpenGL (3.0 or ES 3.2), V4D extends its functionality through bindings to prominent libraries:
- **NanoVG:** For 2D vector and raster graphics, including font rendering.
- **bgfx:** A 3D engine modified to defer its concurrency model to Plan for optimal parallel execution.
- **IMGui:** A lightweight GUI overlay.

V4D encourages direct OpenGL usage and external API integrations via **context sharing**, which is implemented using shared textures. Each external API operates within its isolated OpenGL state machine, maintaining thread safety and modularity.

The runtime’s capabilities are further augmented by its integration with OpenCV, providing:
- **Hardware Acceleration:** Utilizing OpenGL for graphics, VAAPI and NVENC for video, and OpenCL-OpenGL interop for compute tasks.
- **Data Sharing on GPU:** Depending on hardware and software features, V4D can directly share or copy data within GPU memory for efficient processing.

### Integration and Platform Support
V4D and Plan share a tightly bonded design, simplifying combined use cases. However, plans are underway to decouple them, enabling the adoption of alternative runtimes. V4D is actively developed for Linux (X11 and Wayland via EGL or GLX), with auto-detection of supported backends. While macOS support lags slightly, Windows compatibility remains untested but is considered during development.

### Key Principles and Features
1. **Fine-Grained Edge Calls:** Plan introduces specialized edge calls (e.g., `R`, `RW`, `V`) to define data access patterns, supporting smart pointers and OpenCV `UMat` objects. This granularity allows better graph optimization.
2. **State and Data Transparency:** Functions and data in a Plan must avoid introducing hidden states unless explicitly marked as shared. This principle ensures the integrity of the graph and its optimizations.
3. **Parallelism and Pipelining:** Multiple OpenGL contexts can be created and utilized in parallel, making V4D a robust solution for high-performance graphics applications.
4. **Algorithm Modularity:** By structuring algorithms into smaller, reusable fragments or sub-Plans, Plan fosters modular development and scalability.

### Selected Commented Examples (read sequentially)
The following examples have been selected to deepen your understanding of Plan-V4D. There are many more.

#### Blue Sreen using OpenGL
[source](modules/v4d/samples/render_opengl.cpp)

#### Displaying an Image using NanoVG
[source](modules/v4d/samples/display_image_nvg.cpp)

#### A realtime beauty filter (using sub-plans)
[source](modules/v4d/samples/beauty-demo.cpp)

## Why Plan-V4D?

* Computation Graph Engine: Fast parallel code.
* OpenGL: Easy access to OpenGL.
* GUI: Simple yet powerful user interfaces through ImGui.
* Vector graphics: Elegant and fast vector graphics through NanoVG.
* 3D graphics: Powerful 3D graphics through bgfx.
* Font rendering: Loading of fonts and sophisticated rendering options.
* Video pipeline: Through a simple source/sink system videos can be efficently read, displayed, edited and saved.
* Hardware acceleration: Transparent hardware acceleration usage where possible. (e.g. OpenGL, OpenCL, CL-GL interop, VAAPI and CL-VAAPI interop, nvenc). Actually it is possible to write programs that 
* No more highgui with it's heavy dependencies, licenses and limitations.

Please refer to the examples and demos as well as [this OpenCV issue](https://github.com/opencv/opencv/issues/22923) to find out exactly what it can do for you.

## GPU Support
* Intel Gen 8+ (Tested: Gen 11 + Gen 13) tested
* NVIDIA Ada Lovelace (Tested: GTX 4070 Ti) with proprietary drivers (535.104.05) and CUDA toolkit (12.2) tested.
* Intel Arc770 (Mesa 24.3.1) tested
* AMD: never tested

## Requirements
* C++20 (at the moment)
* OpenGL 3.2 Core (optionally Compat)/OpenGL ES 3.0/WebGL2

## Optional requirements
* Support for OpenCL 1.2
* Support for cl_khr_gl_sharing and cl_intel_va_api_media_sharing OpenCL extensions.

## Dependencies
* My OpenCV 4.x fork (It works with mainline OpenCV 4.x as well, but using my fork is highly recommended because it features several improvements and fixes)
* GLEW
* GLFW3
* NanoVG (included as a sub-repo)
* ImGui (included as a sub-repo)
* bgfx (included as a sub-repo)
* Glad (included)

