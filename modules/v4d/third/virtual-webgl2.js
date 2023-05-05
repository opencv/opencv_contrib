/*
 * Copyright 2018, Gregg Tavares.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following disclaimer
 *    in the documentation and/or other materials provided with the
 *    distribution.
 *
 *  * Neither the name of Gregg Tavares. nor the names of his
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/* eslint-env browser */

(function() {
  const settings = {
    disableWebGL1: false,
    compositorCreator() {
    },
  };
  const canvasToVirtualContextMap = new Map();
  const extensionInfo = {};
  const extensionSaveRestoreHelpersArray = [];
  const extensionSaveRestoreHelpers = {};

  let currentVirtualContext = null;
  let someContextsNeedRendering;

  const sharedWebGLContext = document.createElement('canvas').getContext('webgl2');
  const numAttribs = sharedWebGLContext.getParameter(sharedWebGLContext.MAX_VERTEX_ATTRIBS);
  const numTextureUnits = sharedWebGLContext.getParameter(sharedWebGLContext.MAX_COMBINED_TEXTURE_IMAGE_UNITS);
  const numUniformBufferBindings = sharedWebGLContext.getParameter(sharedWebGLContext.MAX_UNIFORM_BUFFER_BINDINGS);
  const baseState = makeDefaultState(sharedWebGLContext, 300, 150);

  const INT = 0x1404
  const UNSIGNED_INT = 0x1405; 
  const FLOAT = 0x1406;

  const vs = `
  attribute vec4 position;
  varying vec2 v_texcoord;
  void main() {
    gl_Position = position;
    v_texcoord = position.xy * .5 + .5;
  }
  `;

  const fs = `
  precision mediump float;
  varying vec2 v_texcoord;
  uniform sampler2D u_tex;
  void main() {
    gl_FragColor = texture2D(u_tex, v_texcoord);
  }
  `;

  const fs2 = `
  precision mediump float;
  varying vec2 v_texcoord;
  uniform sampler2D u_tex;
  void main() {
    gl_FragColor = texture2D(u_tex, v_texcoord);
    gl_FragColor.rgb *= gl_FragColor.a;
  }
  `;

  const premultiplyAlphaTrueProgram = createProgram(sharedWebGLContext, [vs, fs]);
  const premultiplyAlphaFalseProgram = createProgram(sharedWebGLContext, [vs, fs2]);

  {
    const gl = sharedWebGLContext;
    const positionLoc = 0;  // hard coded in createProgram

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
      -1,  1,
       1, -1,
       1,  1,
    ]), gl.STATIC_DRAW);

    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

  }

  saveAllState(baseState);

    HTMLCanvasElement.prototype.getContext = (function(origFn) {
    return function(type, contextAttributes) {
      if (type === 'webgl' || type === 'experimental-webgl') {
        return createOrGetVirtualWebGLContext(this, type, contextAttributes);
      } else if (type === 'webgl2') {
        return createOrGetVirtualWebGLContext(this, type, contextAttributes);
      }
      return origFn.call(this, type, contextAttributes);
    };

  }(HTMLCanvasElement.prototype.getContext));

  function valueOrDefault(value, defaultValue) {
    return value === undefined ? defaultValue : value;
  }

  function errorDisposedContext(fnName) {
    return function() {
      throw new Error(`tried to call ${fnName} on disposed context`);
    };
  }

  class DefaultCompositor {
    constructor(canvas) {
      this._ctx = canvas.getContext('2d');
    }
    composite(gl, texture, canvas, contextAttributes) {
      // note: not entirely sure what to do here. We need this canvas to be at least as large
      // as the canvas we're drawing to. Resizing a canvas is slow so I think just making
      // sure we never get smaller than the largest canvas. At the moment though I'm too lazy
      // to make it smaller.
      const ctx = this._ctx;
      const width = canvas.width;
      const height = canvas.height;
      const maxWidth = Math.max(gl.canvas.width, width);
      const maxHeight = Math.max(gl.canvas.height, height);
      if (gl.canvas.width !== maxWidth || gl.canvas.height !== maxHeight) {
        gl.canvas.width = maxWidth;
        gl.canvas.height = maxHeight;
      }

      gl.viewport(0, 0, width, height);

      gl.useProgram(contextAttributes.premultipliedAlpha ? premultiplyAlphaTrueProgram : premultiplyAlphaFalseProgram);

      // draw the drawingbuffer's texture to the offscreen canvas
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      // copy it to target canvas
      ctx.globalCompositeOperation = 'copy';
      ctx.drawImage(
        gl.canvas,
        0, maxHeight - height, width, height,   // src rect
        0, 0, width, height);  // dest rect
    }
    dispose() {
    }
  }

  function virtualGLConstruct(canvas, contextAttributes = {}, compositor, disposeHelper) {
    const gl = sharedWebGLContext;
    this._canvas = canvas;
    // Should use Symbols or something to hide these variables from the outside.

    this._compositor = compositor;
    this._disposeHelper = disposeHelper;
    this._extensions = {};
    // based on context attributes and canvas.width, canvas.height
    // create a texture and framebuffer
    this._drawingbufferTexture = gl.createTexture();
    this._drawingbufferFramebuffer = gl.createFramebuffer();
    this._contextAttributes = {
      alpha: valueOrDefault(contextAttributes.alpha, true),
      antialias: false,
      depth: valueOrDefault(contextAttributes.depth, true),
      failIfMajorPerformanceCaveat: false,
      premultipliedAlpha: valueOrDefault(contextAttributes.premultipliedAlpha, true),
      stencil: valueOrDefault(contextAttributes.stencil, false),
    };
    this._preserveDrawingbuffer = valueOrDefault(contextAttributes.preserveDrawingBuffer, false);

    const oldTexture = gl.getParameter(gl.TEXTURE_BINDING_2D);
    const oldFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);

    gl.bindTexture(gl.TEXTURE_2D, this._drawingbufferTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    // this._drawingbufferTexture.id = canvas.id;
    // this._drawingbufferFramebuffer.id = canvas.id;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this._drawingbufferFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this._drawingbufferTexture, 0);

    if (this._contextAttributes.depth) {
      const oldRenderbuffer = gl.getParameter(gl.RENDERBUFFER_BINDING);
      this._depthRenderbuffer = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, this._depthRenderbuffer);
      const attachmentPoint = this._contextAttributes.stencil  ? gl.DEPTH_STENCIL_ATTACHMENT : gl.DEPTH_ATTACHMENT;
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, attachmentPoint, gl.RENDERBUFFER, this._depthRenderbuffer);
      gl.bindRenderbuffer(gl.RENDERBUFFER, oldRenderbuffer);
    }

    gl.bindTexture(gl.TEXTURE_2D, oldTexture);
    gl.bindFramebuffer(gl.FRAMEBUFFER, oldFramebuffer);

    // remember all WebGL state (default bindings, default texture units,
    // default attributes and/or vertex shade object, default program,
    // default blend, stencil, zBuffer, culling, viewport etc... state
    this._state = makeDefaultState(gl, canvas.width, canvas.height);
    this._state.readFramebuffer = this._drawingbufferFramebuffer;
    this._state.drawFramebuffer = this._drawingbufferFramebuffer;
    this._state.readBuffer = gl.COLOR_ATTACHMENT0;

    this._state.vertexArray = gl.createVertexArray();
    this._defaultVertexArray = this._state.vertexArray;
  }

  function virtualGLDispose() {
    this._disposeHelper();
    const gl = sharedWebGLContext;
    gl.deleteFramebuffer(this._drawingbufferFramebuffer);
    gl.deleteTexture(this._drawingbufferTexture);
    if (this._depthRenderbuffer) {
      gl.deleteRenderbuffer(this._depthRenderbuffer);
    }
    if (this._compositor.dispose) {
      this._compositor.dispose();
    }
    for (const [key, value] of Object.entries(this)) {
      if (typeof value === 'function') {
        this[key] = errorDisposedContext(key);
      }
    }
    for (const [key, value] of Object.entries(this.prototype)) {
      if (typeof value === 'function') {
        this[key] = errorDisposedContext(key);
      }
    }
  }

  function virtualGLComposite(gl) {
    this._compositor.composite(gl, this._drawingbufferTexture, this.canvas, this._contextAttributes);
    if (!this._preserveDrawingbuffer) {
      this._needClear = true;
    }
  }

  // Base exists so VirtualWebGLContext has a base class we can replace
  // because otherwise it's base is Object which we can't replace.
  class Base {}
  class VirtualWebGLContext extends Base {
    constructor(canvas, contextAttributes = {}, compositor, disposeHelper) {
      super();
      this.dispose = virtualGLDispose;
      this.composite = virtualGLComposite;
      virtualGLConstruct.call(this, canvas, contextAttributes, compositor, disposeHelper);
    }
    get canvas() {
      return this._canvas;
    }
    get drawingBufferWidth() {
      return this.canvas.width;
    }
    get drawingBufferHeight() {
      return this.canvas.height;
    }
  }
  class Base2 {}
  class VirtualWebGL2Context extends Base2 {
    constructor(canvas, contextAttributes = {}, compositor, disposeHelper) {
      super();
      this.dispose = virtualGLDispose;
      this.composite = virtualGLComposite;
      virtualGLConstruct.call(this, canvas, contextAttributes, compositor, disposeHelper);
    }
    get canvas() {
      return this._canvas;
    }
    get drawingBufferWidth() {
      return this.canvas.width;
    }
    get drawingBufferHeight() {
      return this.canvas.height;
    }
  }

  // Replace the prototype with WebGL2RenderingContext so that someCtx instanceof WebGL2RenderingContext returns true
  Object.setPrototypeOf(Object.getPrototypeOf(VirtualWebGLContext.prototype), WebGLRenderingContext.prototype);
  Object.setPrototypeOf(Object.getPrototypeOf(VirtualWebGL2Context.prototype), WebGL2RenderingContext.prototype);

  function makeDefaultState(gl, width, height) {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const state = {
      arrayBuffer: null,
      renderbuffer: null,
      drawFramebuffer: null,
      readFramebuffer: null,
      copyReadBuffer: null,
      copyWriteBuffer: null,
      pixelPackBuffer: null,
      pixelUnpackBuffer: null,
      transformFeedbackBuffer: null,
      uniformBuffer: null,

      readBuffer: gl.BACK,

      enable: new Map([
        [ gl.BLEND, false ],
        [ gl.CULL_FACE, false ],
        [ gl.DEPTH_TEST, false ],
        [ gl.DITHER, false ],
        [ gl.POLYGON_OFFSET_FILL, false ],
        [ gl.RASTERIZER_DISCARD, false ],
        [ gl.SAMPLE_ALPHA_TO_COVERAGE, false ],
        [ gl.SAMPLE_COVERAGE, false ],
        [ gl.SCISSOR_TEST, false ],
        [ gl.STENCIL_TEST, false ],
      ]),

      // This is a place the spec gets wrong! This data should have been part of a VertexArray
      attribValues: new Array(numAttribs).fill(0).map(() => {
        return {
          type: gl.FLOAT,
          value: [0, 0, 0, 1],
        };
      }),

      vertexArray: vao,
      activeTexture: gl.TEXTURE0,
      transformFeedback: null,

      pack: new Map([
        [ gl.PACK_ALIGNMENT, 4],
        [ gl.UNPACK_ALIGNMENT, 4],
        [ gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.BROWSER_DEFAULT_WEBGL],
        [ gl.UNPACK_FLIP_Y_WEBGL, 0],
        [ gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 0],
        [ gl.UNPACK_ROW_LENGTH   , 0],
        [ gl.UNPACK_SKIP_ROWS    , 0],
        [ gl.UNPACK_SKIP_PIXELS  , 0],
        [ gl.UNPACK_SKIP_IMAGES  , 0],
        [ gl.UNPACK_IMAGE_HEIGHT , 0],
        [ gl.PACK_ROW_LENGTH     , 0],
        [ gl.PACK_SKIP_ROWS      , 0],
        [ gl.PACK_SKIP_PIXELS    , 0],
      ]),

      currentProgram: null,
      viewport: [0, 0, width, height],
      scissor: [0, 0, width, height],
      blendSrcRgb: gl.ONE,
      blendDstRgb: gl.ZERO,
      blendSrcAlpha: gl.ONE,
      blendDstAlpha: gl.ZERO,
      blendEquationRgb: gl.FUNC_ADD,
      blendEquationAlpha: gl.FUNC_ADD,
      blendColor: [0, 0, 0, 0],
      clearColor: [0, 0, 0, 0],
      colorMask: [true, true, true, true],
      cullFaceMode: gl.BACK,
      clearDepth: 1,
      depthFunc: gl.LESS,
      depthRange: [0, 1],
      depthMask: true,
      frontFace: gl.CCW,
      generateMipmapHint: gl.DONT_CARE,
      lineWidth: 1,
      polygonOffsetFactor: 0,
      polygonOffsetUnits: 0,
      sampleCoverageValue: 1,
      sampleCoverageUnits: false,
      stencilFront: {
        fail: gl.KEEP,
        func: gl.ALWAYS,
        depthFail: gl.KEEP,
        depthPass: gl.KEEP,
        ref: 0,
        valueMask: 0xFFFFFFFF,
        writeMask: 0xFFFFFFFF,
      },
      stencilBack: {
        fail: gl.KEEP,
        func: gl.ALWAYS,
        depthFail: gl.KEEP,
        depthPass: gl.KEEP,
        ref: 0,
        valueMask: 0xFFFFFFFF,
        writeMask: 0xFFFFFFFF,
      },
      stencilClearValue: 0,

      textureUnits: new Array(numTextureUnits).fill(0).map(() => {
        return {
          texture2D: null,
          textureCubemap: null,
          texture2DArray: null,
          texture3D: null,
          sampler: null,
        };
      }),
      uniformBufferBindings: new Array(numUniformBufferBindings).fill(0).map(() => {
        return {
          buffer: null,
          size: 0,
          start: 0,
        };
      }),
    };

    return state;
  }

  function isFramebufferBindingNull(vCtx) {
    return vCtx._state.drawFramebuffer === vCtx._drawingbufferFramebuffer;
  }

  function createWrapper(origFn/*, name*/) {
    // lots of optimization could happen here depending on specific functions
    return function(...args) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      return origFn.call(sharedWebGLContext, ...args);
    };
  }

  function clearIfNeeded(vCtx) {
    if (vCtx._needClear) {
      vCtx._needClear = false;
      const gl = sharedWebGLContext;
      gl.bindFramebuffer(gl.FRAMEBUFFER, vCtx._drawingbufferFramebuffer);
      gl.disable(gl.SCISSOR_TEST);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
      enableDisable(gl, gl.SCISSOR_TEST, vCtx._state.scissorTest);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, vCtx._state.drawFramebuffer);
    }
  }

  function beforeDraw(vCtx) {
    makeCurrentContext(vCtx);
    resizeCanvasIfChanged(vCtx);
    clearIfNeeded(vCtx);
  }

  function afterDraw(vCtx) {
    if (isFramebufferBindingNull(vCtx)) {
      vCtx._needComposite = true;
      if (!someContextsNeedRendering) {
        someContextsNeedRendering = true;
        setTimeout(renderAllDirtyVirtualCanvases, 0);
      }
    }
  }

  function createDrawWrapper(origFn) {
    return function(...args) {
      // a rendering function was called so we need to copy are drawingBuffer
      // to the canvas for this context after the current event.
      beforeDraw(this);
      const result = origFn.call(sharedWebGLContext, ...args);
      afterDraw(this);
      return result;
    };
  }

  function createStateArgsSaverFn(fnName) {
    return function(...args) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl[fnName](...args);
      const v = this._state[fnName];
      for (let i = 0; i < args.length; ++i) {
        v[i] = args[i];
      }
    };
  }

  function createSaveStateNamedArgs(fnName, argsToStateProps) {
    return function(...args) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl[fnName](...args);
      for (let i = 0; i < argsToStateProps.length; ++i) {
        this._state[argsToStateProps[i]] = args[i];
      }
    };
  }

  function createVertexAttribWrapper(origFn, fn) {
    return function(loc, ...args) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      origFn.call(gl, loc, ...args);
      const [type, value] = fn(args);
      this._state.attribValues[loc] = {type, value};
    };
  }

  function saveStencilMaskImpl(st, mask) {
    st.writeMask = mask;
  }

  function saveStencilMask(state, face, mask) {
    if (face === sharedWebGLContext.FRONT || face === sharedWebGLContext.FRONT_AND_BACK) {
      saveStencilMaskImpl(state.stencilFront, mask);
    }
    if (face === sharedWebGLContext.BACK || face === sharedWebGLContext.FRONT_AND_BACK) {
      saveStencilMaskImpl(state.stencilBack, mask);
    }
  }

  function saveStencilFuncImpl(st, func, ref, mask) {
    st.func = func;
    st.ref = ref;
    st.valueMask = mask;
  }

  function saveStencilFunc(state, face, func, ref, mask) {
    if (face === sharedWebGLContext.FRONT || face === sharedWebGLContext.FRONT_AND_BACK) {
      saveStencilFuncImpl(state.stencilFront, func, ref, mask);
    }
    if (face === sharedWebGLContext.BACK || face === sharedWebGLContext.FRONT_AND_BACK) {
      saveStencilFuncImpl(state.stencilBack, func, ref, mask);
    }
  }

  function saveStencilOpImpl(st, fail, zfail, zpass) {
    st.fail = fail;
    st.depthFail = zfail;
    st.depthPass = zpass;
  }

  function saveStencilOp(state, face, fail, zfail, zpass) {
    if (face === sharedWebGLContext.FRONT || face === sharedWebGLContext.FRONT_AND_BACK) {
      saveStencilOpImpl(state.stencilFront, fail, zfail, zpass);
    }
    if (face === sharedWebGLContext.BACK || face === sharedWebGLContext.FRONT_AND_BACK) {
      saveStencilOpImpl(state.stencilBack, fail, zfail, zpass);
    }
  }

  const virtualFns = {
    getExtension(name) {
      // just like the real context each extension needs a virtual class because each use
      // of the extension might be modified (as in people adding properties to it)
      const existingExt = this._extensions[name];
      if (existingExt) {
        return existingExt;
      }

      const ext = sharedWebGLContext.getExtension(name);
      if (!ext) {
        return null;
      }
      const wrapperInfo = extensionInfo[name] || {};
      const wrapperFnMakerFn = wrapperInfo.wrapperFnMakerFn || (() => {
        console.log('trying to get extension:', name);
      });
      const saveRestoreHelper = extensionSaveRestoreHelpers[name];
      if (!saveRestoreHelper) {
        const saveRestoreMakerFn = wrapperInfo.saveRestoreMakerFn;
        if (saveRestoreMakerFn) {
          const saveRestore = saveRestoreMakerFn(ext);
          extensionSaveRestoreHelpers[name] = saveRestore;
          extensionSaveRestoreHelpersArray.push(saveRestore);
        }
      }

      const wrapper = {
        _context: this,
      };
      for (const key in ext) {
        let value = ext[key];
        if (typeof value === 'function') {
          value = wrapperFnMakerFn(ext, value, name);
        }
        wrapper[key] = value;
      }

      return wrapper;
    },
    activeTexture(unit) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.activeTexture(unit);
      this._state.activeTexture = unit;
    },
    enable(pname) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.enable(pname);
      this._state.enable.set(pname, true);
    },
    disable(pname) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.disable(pname);
      this._state.enable.set(pname, false);
    },
    viewport: createStateArgsSaverFn('viewport'),
    scissor: createStateArgsSaverFn('scissor'),
    blendColor: createStateArgsSaverFn('blendColor'),
    clearColor: createStateArgsSaverFn('clearColor'),
    colorMask: createStateArgsSaverFn('colorMask'),
    depthRange: createStateArgsSaverFn('depthRange'),
    bindBuffer(target, buffer) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.bindBuffer(target, buffer);
      switch (gl.target) {
        case gl.ARRAY_BUFFER:
          this._state.arrayBuffer = buffer;
          break;
        case gl.COPY_READ_BUFFER:
          this._state.copyReadBuffer = buffer;
          break;
        case gl.COPY_WRITE_BUFFER:
          this._state.copyWriteBuffer = buffer;
          break;
        case gl.PIXEL_PACK_BUFFER:
          this._state.pixelPackBuffer = buffer;
          break;
        case gl.PIXEL_UNPACK_BUFFER:
          this._state.pixelUnpackBuffer = buffer;
          break;
        case gl.TRANSFORM_FEEDBACK_BUFFER:
          this._state.transformFeedbackBuffer = buffer;
          break;
        case gl.UNIFORM_BUFFER:
          this._state.uniformBuffer = buffer;
      }
    },
    bindBufferBase(target, index, buffer) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.bindBufferBase(target, index, buffer);
      switch (target) {
        case gl.UNIFORM_BUFFER: {
          const ub = this._state.uniformBufferBindings[index];
          ub.buffer = buffer;
          ub.size = 0;
          ub.start = 0;
          break;
        }
        default:
          break;
      }
    },
    bindBufferRange(target, index, buffer, offset, size) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.bindBufferRange(target, index, buffer, offset, size);
      switch (target) {
        case gl.UNIFORM_BUFFER: {
          const ub = this._state.uniformBufferBindings[index];
          ub.buffer = buffer;
          ub.size = size;
          ub.start = offset;
          break;
        }
        default:
          break;
      }
    },
    bindTexture(target, texture) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.bindTexture(target, texture);
      const unit = this._state.textureUnits[this._state.activeTexture - gl.TEXTURE0];
      switch (target) {
        case gl.TEXTURE_2D:
          unit.texture2D = texture;
          break;
        case gl.TEXTURE_CUBE_MAP:
          unit.textureCubemap = texture;
          break;
        case gl.TEXTURE_2D_ARRAY:
          unit.texture2DArray = texture;
          break;
        case gl.TEXTURE_3D:
          unit.texture3D = texture;
          break;
      }
    },
    bindRenderbuffer(target, renderbuffer) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.bindRenderbuffer(target, renderbuffer);
      this._state.renderbuffer = renderbuffer;
    },
    bindSampler(unit, sampler) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.bindSampler(unit, sampler);
      this._state.textureUnits[unit].sampler = sampler;
    },
    bindVertexArray(va) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      if (va === null) {
        va = this._defaultVertexArray;
      }
      gl.bindVertexArray(va);
      this._state.vertexArray = va;
    },
    getContextAttributes() {
      return this._contextAttributes;
    },
    readPixels(...args) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      clearIfNeeded(this);
      const gl = sharedWebGLContext;
      return gl.readPixels(...args);
    },
    getParameter(pname) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      const value = gl.getParameter(pname);
      switch (pname) {
        case gl.FRAMEBUFFER_BINDING:
          if (value === this._drawingbufferFramebuffer) {
            return null;
          }
          break;
        case gl.DRAW_BUFFER0:
          if (isFramebufferBindingNull(this)) {
            if (value === gl.COLOR_ATTACHMENT0) {
              return gl.BACK;
            }
          }
          break;
        case gl.READ_BUFFER:
          if (isFramebufferBindingNull(this)) {
            if (value === gl.COLOR_ATTACHMENT0) {
              return gl.BACK;
            }
          }
          break;
        case gl.VERTEX_ARRAY_BINDING:
          if (value === this._defaultVertexArray) {
            return null;
          }
          break;
      }
      return value;
    },
    blendFunc(sfactor, dfactor) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.blendFunc(sfactor, dfactor);
      this._state.blendSrcRgb = sfactor;
      this._state.blendSrcAlpha = sfactor;
      this._state.blendDstRgb = dfactor;
      this._state.blendDstAlpha = dfactor;
    },
    blendEquation(mode) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.blendEquation(mode);
      this._state.blendEquationRgb = mode;
      this._state.blendEquationAlpha = mode;
    },
    blendFuncSeparate: createSaveStateNamedArgs('blendFuncSeparate', ['blendSrcRgb', 'blendDstRgb', 'blendSrcAlpha', 'blendDstAlpha']),
    blendEquationSeparate: createSaveStateNamedArgs('blendEquationSeparate', ['blendEquationRgb', 'blendEquationAlpha']),
    cullFace: createSaveStateNamedArgs('cullFace', ['cullFaceMode']),
    clearDepth: createSaveStateNamedArgs('clearDepth', ['clearDepth']),
    depthFunc: createSaveStateNamedArgs('depthFunc', ['depthFunc']),
    depthMask: createSaveStateNamedArgs('depthMask', ['depthMask']),
    frontFace: createSaveStateNamedArgs('frontFace', ['frontFace']),
    lineWidth: createSaveStateNamedArgs('lineWidth', ['lineWidth']),
    polygonOffset: createSaveStateNamedArgs('polygonOffset', ['polygonOffsetFactor', 'polygonOffsetUnits']),
    sampleCoverage: createSaveStateNamedArgs('sampleCoverage', ['sampleCoverageValue', 'sampleCoverageUnits']),
    clearStencil: createSaveStateNamedArgs('clearStencil', ['clearStencilValue']),
    hint(pname, value) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.hint(pname, value);
      this._state.generateMipmapHint = value;
    },
    bindFramebuffer(bindPoint, framebuffer) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      if (framebuffer === null) {
        // bind our drawingBuffer
        framebuffer = this._drawingbufferFramebuffer;
      }
      gl.bindFramebuffer(bindPoint, framebuffer);
      switch (bindPoint) {
        case gl.FRAMEBUFFER:
          this._state.readFramebuffer = framebuffer;
          this._state.drawFramebuffer = framebuffer;
          break;
        case gl.DRAW_FRAMEBUFFER:
          this._state.drawFramebuffer = framebuffer;
          break;
        case gl.READ_FRAMEBUFFER:
          this._state.readFramebuffer = framebuffer;
          break;
      }
    },
    drawBuffers: (function() {
      const gl = sharedWebGLContext;
      const backBuffer = [gl.COLOR_ATTACHMENT0];

      return function(drawingBuffers) {
        makeCurrentContext(this);
        resizeCanvasIfChanged(this);
        // if the virtual context is bound to canvas then fake it
        if (isFramebufferBindingNull(this)) {
          // this really isn't checking everything
          // for example if the user passed in array.length != 1
          // then we are supposed to generate an error
          if (drawingBuffers[0] === gl.BACK) {
            drawingBuffers = backBuffer;
          }
        }

        gl.drawBuffers(drawingBuffers);
      };
    }()),
    clear: createDrawWrapper(WebGL2RenderingContext.prototype.clear),
    drawArrays: createDrawWrapper(WebGL2RenderingContext.prototype.drawArrays),
    drawElements: createDrawWrapper(WebGL2RenderingContext.prototype.drawElements),
    drawArraysInstanced: createDrawWrapper(WebGL2RenderingContext.prototype.drawArraysInstanced),
    drawElementsInstanced: createDrawWrapper(WebGL2RenderingContext.prototype.drawElementsInstanced),
    drawRangeElements: createDrawWrapper(WebGL2RenderingContext.prototype.drawRangeElements),
    useProgram(program) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.useProgram(program);
      this._state.currentProgram = program;
    },
    bindTransformFeedback(target, tb) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.bindTransformFeedback(target, tb);
      this._state.transformFeedback = tb;
    },
    readBuffer(src) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      if (src === gl.BACK) {
        src = gl.COLOR_ATTACHMENT0;
      }
      gl.readBuffer(src);
      this._state.readBuffer = src;
    },
    stencilFunc(func, ref, mask) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.stencilFunc(func, ref, mask);
      saveStencilFunc(this._state, gl.FRONT_AND_BACK, func, ref, mask);
    },
    stencilFuncSeparate(face, func, ref, mask) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.stencilFuncSeparate(face, func, ref, mask);
      saveStencilFunc(this._state, face, func, ref, mask);
    },
    stencilMask(mask) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.stencilMask(mask);
      saveStencilMask(this._state, gl.FRONT_AND_BACK, mask);
    },
    stencilMaskSeparate(face, mask) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.stencilMaskSeparate(face, mask);
      saveStencilMask(this._state, face, mask);
    },
    stencilOp(fail, zfail, zpass) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.stencilOp(fail, zfail, zpass);
      saveStencilOp(this._state, gl.FRONT_AND_BACK, fail, zfail, zpass);
    },
    stencilOpSeparate(face, fail, zfail, zpass) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      gl.stencilOpSeparate(face, fail, zfail, zpass);
      saveStencilOp(this._state, face, fail, zfail, zpass);
    },
    vertexAttrib1f:   createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1f,   ([x])            => [FLOAT, [x, 0, 0, 1]]),
    vertexAttrib2f:   createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1f,   ([x, y])         => [FLOAT, [x, y, 0, 1]]),
    vertexAttrib3f:   createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1f,   ([x, y, z])      => [FLOAT, [x, y, z, 1]]),
    vertexAttrib4f:   createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1f,   ([x, y, z, w])   => [FLOAT, [x, y, z, w]]),
    vertexAttrib1fv:  createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1fv,  ([[x]])          => [FLOAT, [x, 0, 0, 1]]),
    vertexAttrib2fv:  createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1fv,  ([[x, y]])       => [FLOAT, [x, y, 0, 1]]),
    vertexAttrib3fv:  createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1fv,  ([[x, y, z]])    => [FLOAT, [x, y, z, 1]]),
    vertexAttrib4fv:  createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1fv,  ([[x, y, z, w]]) => [FLOAT, [x, y, z, w]]),
    vertexAttrib1i:   createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1i,   ([x])            => [FLOAT, [x, 0, 0, 1]]),
    vertexAttrib2i:   createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1i,   ([x, y])         => [FLOAT, [x, y, 0, 1]]),
    vertexAttrib3i:   createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1i,   ([x, y, z])      => [FLOAT, [x, y, z, 1]]),
    vertexAttrib4i:   createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1i,   ([x, y, z, w])   => [FLOAT, [x, y, z, w]]),
    vertexAttrib1iv:  createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1iv,  ([[x]])          => [FLOAT, [x, 0, 0, 1]]),
    vertexAttrib2iv:  createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1iv,  ([[x, y]])       => [FLOAT, [x, y, 0, 1]]),
    vertexAttrib3iv:  createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1iv,  ([[x, y, z]])    => [FLOAT, [x, y, z, 1]]),
    vertexAttrib4iv:  createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1iv,  ([[x, y, z, w]]) => [FLOAT, [x, y, z, w]]),
    vertexAttrib1ui:  createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1ui,  ([x])            => [FLOAT, [x, 0, 0, 1]]),
    vertexAttrib2ui:  createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1ui,  ([x, y])         => [FLOAT, [x, y, 0, 1]]),
    vertexAttrib3ui:  createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1ui,  ([x, y, z])      => [FLOAT, [x, y, z, 1]]),
    vertexAttrib4ui:  createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1ui,  ([x, y, z, w])   => [FLOAT, [x, y, z, w]]),
    vertexAttrib1uiv: createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1uiv, ([[x]])          => [FLOAT, [x, 0, 0, 1]]),
    vertexAttrib2uiv: createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1uiv, ([[x, y]])       => [FLOAT, [x, y, 0, 1]]),
    vertexAttrib3uiv: createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1uiv, ([[x, y, z]])    => [FLOAT, [x, y, z, 1]]),
    vertexAttrib4uiv: createVertexAttribWrapper(WebGL2RenderingContext.prototype.vertexAttrib1uiv, ([[x, y, z, w]]) => [FLOAT, [x, y, z, w]]),
  };

  const webgl1Extensions = {
    OES_texture_float: {
      fn() {
        return {};
      },
    },
    OES_vertex_array_object: {
      fn(vCtx) {
        return {
          VERTEX_ARRAY_BINDING_OES: sharedWebGLContext.VERTEX_ARRAY_BINDING,
          createVertexArrayOES() {
            return sharedWebGLContext.createVertexArray();
          },
          deleteVertexArrayOES(va) {
            sharedWebGLContext.deleteVertexArray(va);
          },
          bindVertexArrayOES(va) {
            virtualFns.bindVertexArray.call(vCtx, va);
          },
          isVertexArrayOES(va) {
            return sharedWebGLContext.isVertexArray(va);
          },
        };
      },
    },
    ANGLE_instanced_arrays: {
      fn(vCtx) {
        return {
          VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE: 0x88FE,
          drawArraysInstancedANGLE(...args) {
            virtualFns.drawArraysInstanced.call(vCtx, ...args);
          },
          drawElementsInstancedANGLE(...args) {
            virtualFns.drawElementsInstanced.call(vCtx, ...args);
          },
          vertexAttribDivisorANGLE(...args) {
            sharedWebGLContext.vertexAttribDivisor(...args);
          },
        };
      },
    },
    // We can't easily support WebGL_draw_buffers because WebGL2 does not
    // support gl_FragData. Instead, it requires you to write your shaders
    // in GLSL ES 3.0 if you want to use multiple color attachments. To support
    // that correctly would require writing a GLSL parser. We need to change
    // 'attribute' -> 'in', 'varying' -> 'out' in vertex shaders, 'varying' to 'in'
    // in fragment shader, 'texture2D' to 'texture' and declare outputs. That sounds
    // simple but it quickly becomes complicated.
    //
    // * 'texture' is a valid identifier in GLSL ES 1.0 but a reserved word in
    //   GLSL ES 3.0 so we'd have to rename identifiers
    //
    // * Changing the fragment shader means it's on longer compatible with
    //   GLSL ES 1.0 vertex shaders. But, back in WebGL1, we could easily use
    //   the same vertex shader with and without WEBGL_draw_buffers. That means
    //   we now need 2 versions of every vertex shader (ES 1.0 and ES 3.0), OR
    //   we need to translate ALL shaders to GLSL ES 3.0
    //
    // * GLSL 1.0 shaders support dynamically indexing an array of samplers.
    //   GLSL 3.0 does not. So we'd have to emit an emulation function.
    //
    // The point is it's not a trivial amount of work.
    /*
    WEBGL_draw_buffers: {
      fn(vCtx) {
        return {
          COLOR_ATTACHMENT0_WEBGL     : 0x8CE0,
          COLOR_ATTACHMENT1_WEBGL     : 0x8CE1,
          COLOR_ATTACHMENT2_WEBGL     : 0x8CE2,
          COLOR_ATTACHMENT3_WEBGL     : 0x8CE3,
          COLOR_ATTACHMENT4_WEBGL     : 0x8CE4,
          COLOR_ATTACHMENT5_WEBGL     : 0x8CE5,
          COLOR_ATTACHMENT6_WEBGL     : 0x8CE6,
          COLOR_ATTACHMENT7_WEBGL     : 0x8CE7,
          COLOR_ATTACHMENT8_WEBGL     : 0x8CE8,
          COLOR_ATTACHMENT9_WEBGL     : 0x8CE9,
          COLOR_ATTACHMENT10_WEBGL    : 0x8CEA,
          COLOR_ATTACHMENT11_WEBGL    : 0x8CEB,
          COLOR_ATTACHMENT12_WEBGL    : 0x8CEC,
          COLOR_ATTACHMENT13_WEBGL    : 0x8CED,
          COLOR_ATTACHMENT14_WEBGL    : 0x8CEE,
          COLOR_ATTACHMENT15_WEBGL    : 0x8CEF,

          DRAW_BUFFER0_WEBGL          : 0x8825,
          DRAW_BUFFER1_WEBGL          : 0x8826,
          DRAW_BUFFER2_WEBGL          : 0x8827,
          DRAW_BUFFER3_WEBGL          : 0x8828,
          DRAW_BUFFER4_WEBGL          : 0x8829,
          DRAW_BUFFER5_WEBGL          : 0x882A,
          DRAW_BUFFER6_WEBGL          : 0x882B,
          DRAW_BUFFER7_WEBGL          : 0x882C,
          DRAW_BUFFER8_WEBGL          : 0x882D,
          DRAW_BUFFER9_WEBGL          : 0x882E,
          DRAW_BUFFER10_WEBGL         : 0x882F,
          DRAW_BUFFER11_WEBGL         : 0x8830,
          DRAW_BUFFER12_WEBGL         : 0x8831,
          DRAW_BUFFER13_WEBGL         : 0x8832,
          DRAW_BUFFER14_WEBGL         : 0x8833,
          DRAW_BUFFER15_WEBGL         : 0x8834,

          MAX_COLOR_ATTACHMENTS_WEBGL : 0x8CDF,
          MAX_DRAW_BUFFERS_WEBGL      : 0x8824,

          drawBuffersWEBGL(buffers) {
            virtualFns.drawBuffers.call(vCtx, buffers);
          },
        };
      },
    },
    */
  };

  const texImage2DArgParersMap = new Map([
    [9, function([target, level, internalFormat, width, height, , format, type]) {
      return {target, level, internalFormat, width, height, format, type};
    }, ],
    [6, function([target, level, internalFormat, format, type, texImageSource]) {
      return {target, level, internalFormat, width: texImageSource.width, height: texImageSource.height, format, type};
    }, ],
    [10, function([target, level, internalFormat, width, height, , format, type]) {
      return {target, level, internalFormat, width, height, format, type};
    }, ],
  ]);

  const webgl1Fns = {
    getExtension(name) {
      const existingExt = this._extensions[name];
      if (existingExt) {
        return existingExt;
      }

      const info = webgl1Extensions[name];
      if (!info) {
        return virtualFns.getExtension.call(this, name);
      }

      return info.fn(this);
    },
    texImage2D(...args) {
      makeCurrentContext(this);
      resizeCanvasIfChanged(this);
      const gl = sharedWebGLContext;
      const fn = texImage2DArgParersMap.get(args.length);
      const {internalFormat, type} = fn(args);
      if (type === sharedWebGLContext.FLOAT) {
        switch (internalFormat) {
          case gl.RGBA: args[2] = gl.RGBA32F; break;
          case gl.RGB: args[2] = gl.RGB32F; break;
        }
      }
      gl.texImage2D(...args);
    },
    getSupportedExtensions: function() {
      return [
        ...sharedWebGLContext.getSupportedExtensions(),
        'OES_texture_float',
        'WEBGL_depth_texture',
        'OES_vertex_array_object',
        // "WEBGL_draw_buffers",  // See other comment
      ];
    },
  };

  // copy all WebGL constants and functions to the prototype of
  // VirtualWebGLContext
  function copyProperties(keys, VirtualClass, overrideFns) {
    for (const key of keys) {
      const propDesc = Object.getOwnPropertyDescriptor(WebGL2RenderingContext.prototype, key);
      if (propDesc.get) {
        // it's a getter/setter ?
        const virtualPropDesc = Object.getOwnPropertyDescriptor(VirtualClass.prototype, key);
        if (!virtualPropDesc) {
          console.warn(`WebGL2RenderingContext.${key} is not supported`);
        }
        continue;
      }
      switch (key) {
        default: {
          const value = WebGL2RenderingContext.prototype[key];
          let newValue = value;
          const fn = overrideFns[key] || virtualFns[key];
          if (fn) {
            newValue = fn;
          } else {
            if (typeof value === 'function') {
              newValue = createWrapper(value, key);
            }
          }
           VirtualClass.prototype[key] = newValue;
           break;
        }
      }
    }
  }
  copyProperties(Object.keys(WebGLRenderingContext.prototype), VirtualWebGLContext, webgl1Fns);
  copyProperties(Object.keys(WebGL2RenderingContext.prototype), VirtualWebGL2Context, {});

  function makeCurrentContext(vCtx) {
    if (currentVirtualContext === vCtx) {
      return;
    }

    // save all current WebGL state on the previous current virtual context
    if (currentVirtualContext) {
      saveAllState(currentVirtualContext._state, currentVirtualContext);
    }

    // restore all state for the new context
    restoreAllState(vCtx._state, vCtx);

    // check if the current state is supposed to be rendering to the canvas.
    // if so bind vCtx._drawingbuffer

    currentVirtualContext = vCtx;
  }

  function resizeCanvasIfChanged(vCtx) {
    const width = vCtx.canvas.width;
    const height = vCtx.canvas.height;

    if (width !== vCtx._width || height !== vCtx._height) {
      vCtx._width = width;
      vCtx._height = height;
      const gl = sharedWebGLContext;
      const oldTexture = gl.getParameter(gl.TEXTURE_BINDING_2D);
      const format = vCtx._contextAttributes.alpha ? gl.RGBA : gl.RGB;
      gl.bindTexture(gl.TEXTURE_2D, vCtx._drawingbufferTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, format, width, height, 0, format, gl.UNSIGNED_BYTE, null);
      gl.bindTexture(gl.TEXTURE_2D, oldTexture);

      if (vCtx._depthRenderbuffer) {
        const oldRenderbuffer = gl.getParameter(gl.RENDERBUFFER_BINDING);
        const internalFormat = vCtx._contextAttributes.stencil ? gl.DEPTH_STENCIL : gl.DEPTH_COMPONENT16;
        gl.bindRenderbuffer(gl.RENDERBUFFER, vCtx._depthRenderbuffer);
        gl.renderbufferStorage(gl.RENDERBUFFER, internalFormat, width, height);
        gl.bindRenderbuffer(gl.RENDERBUFFER, oldRenderbuffer);
      }
    }
  }

  function createOrGetVirtualWebGLContext(canvas, type, contextAttributes) {
    // check if this canvas already has a context
    const existingVirtualCtx = canvasToVirtualContextMap.get(canvas);
    if (existingVirtualCtx) {
      return existingVirtualCtx;
    }

    const compositor = settings.compositorCreator(canvas, type, contextAttributes) || new DefaultCompositor(canvas, type, contextAttributes);
    const newVirtualCtx = type === 'webgl2'
        ? new VirtualWebGL2Context(canvas, contextAttributes, compositor, () => {
            canvasToVirtualContextMap.delete(canvas);
          })
        : new VirtualWebGLContext(canvas, contextAttributes, compositor, () => {
            canvasToVirtualContextMap.delete(canvas);
          });

    canvasToVirtualContextMap.set(canvas, newVirtualCtx);

    return newVirtualCtx;
  }

  function createProgram(gl, shaderSources) {
    const program = gl.createProgram();
    [gl.VERTEX_SHADER, gl.FRAGMENT_SHADER].forEach((type, ndx) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, shaderSources[ndx]);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(shader)); // eslint-disable-line
      }
      gl.attachShader(program, shader);
    });
    gl.bindAttribLocation(program, 0, 'position');
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(program)); // eslint-disable-line
    }

    return program;
  }

  function saveAllState(state, vCtx) {
    // save all WebGL state (current bindings, current texture units,
    // current attributes and/or vertex shade object, current program,
    // current blend, stencil, zBuffer, culling, viewport etc... state
    for (const fns of extensionSaveRestoreHelpersArray) {
      fns.save(state, vCtx);
    }
  }

  function setStencil(gl, face, st) {
    gl.stencilFuncSeparate(face, st.func, st.ref, st.valueMask);
    gl.stencilOpSeparate(face, st.fail, st.depthFail, st.depthPass);
    gl.stencilMaskSeparate(face, st.writeMask);
  }

  function restoreAllState(state, vCtx) {
    // restore all WebGL state (current bindings, current texture units,
    // current attributes and/or vertex shade object, current program,
    // current blend, stencil, zBuffer, culling, viewport etc... state
    // save all WebGL state (current bindings, current texture units,
    // current attributes and/or vertex shade object, current program,
    // current blend, stencil, zBuffer, culling, viewport etc... state
    const gl = sharedWebGLContext;

    gl.bindRenderbuffer(gl.RENDERBUFFER, state.renderbuffer);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, state.readFramebuffer);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, state.drawFramebuffer);

    // restore attributes
    gl.bindVertexArray(state.vertexArray);
    for (let i = 0; i < numAttribs; ++i) {
      const attr = state.attribValues[i];
      switch (attr.type) {
        case gl.FLOAT:
          gl.vertexAttrib4fv(i, attr.value);
          break;
        case gl.INT:
          gl.vertexAttribI4iv(i, attr.value);
          break;
        case gl.UNSIGNED_INT:
          gl.vertexAttribI4uiv(i, attr.value);
          break;
      }
    }

    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, state.transformFeedback);

    // restore texture units
    for (let i = 0; i < numTextureUnits; ++i) {
      gl.activeTexture(gl.TEXTURE0 + i);
      const unit = state.textureUnits[i];
      gl.bindTexture(gl.TEXTURE_2D, unit.texture2D);
      gl.bindTexture(gl.TEXTURE_CUBE_MAP, unit.textureCubemap);
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, unit.texture2DArray);
      gl.bindTexture(gl.TEXTURE_3D, unit.texture3D);
      gl.bindSampler(i, unit.sampler);
    }
    gl.activeTexture(state.activeTexture);

    // uniform buffer bindings (must be restored before UNIFORM_BUFFER restore)
    for (let i = 0; i < numUniformBufferBindings; ++i) {
      const ub = state.uniformBufferBindings[i];
      if (ub.size || ub.start) {
        gl.bindBufferRange(gl.UNIFORM_BUFFER, i, ub.buffer, ub.start, ub.size);
      } else {
        gl.bindBufferBase(gl.UNIFORM_BUFFER, i, ub.buffer);
      }
    }

    // bindings
    gl.bindBuffer(gl.ARRAY_BUFFER, state.arrayBuffer);
    gl.bindBuffer(gl.COPY_READ_BUFFER, state.copyReadBuffer);
    gl.bindBuffer(gl.COPY_WRITE_BUFFER, state.copyWriteBuffer);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, state.pixelPackBuffer);
    gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, state.pixelUnpackBuffer);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, state.transformFeedbackBuffer);
    gl.bindBuffer(gl.UNIFORM_BUFFER, state.uniformBuffer);

    gl.readBuffer(state.readBuffer);

    state.enable.forEach((value, key) => {
      enableDisable(gl, key, value);
    });

    state.pack.forEach((value, key) => {
      gl.pixelStorei(key, value);
    });

    gl.useProgram(state.currentProgram);

    gl.viewport(...state.viewport);
    gl.scissor(...state.scissor);
    gl.blendFuncSeparate(state.blendSrcRgb, state.blendDstRgb, state.blendSrcAlpha, state.blendDstAlpha);
    gl.blendEquationSeparate(state.blendEquationRgb, state.blendEquationAlpha);
    gl.blendColor(...state.blendColor);
    gl.clearColor(...state.clearColor);
    gl.colorMask(...state.colorMask);
    gl.cullFace(state.cullFaceMode);
    gl.clearDepth(state.clearDepth);
    gl.depthFunc(state.depthFunc);
    gl.depthRange(...state.depthRange);
    gl.depthMask(state.depthMask);
    gl.frontFace(state.frontFace);
    gl.hint(gl.GENERATE_MIPMAP_HINT, state.generateMipmapHint);
    gl.lineWidth(state.lineWidth);
    gl.polygonOffset(state.polygonOffsetFactor, state.polygonOffsetUnits);
    gl.sampleCoverage(state.sampleCoverageValue, state.sampleCoverageUnits);

    setStencil(gl, gl.BACK, state.stencilBack);
    setStencil(gl, gl.FRONT, state.stencilFront);

    gl.clearStencil(state.stencilClearValue);

    for (const fns of extensionSaveRestoreHelpersArray) {
      fns.restore(state, vCtx);
    }
  }

  function enableDisable(gl, feature, enable) {
    if (enable) {
      gl.enable(feature);
    } else {
      gl.disable(feature);
    }
  }

  function renderAllDirtyVirtualCanvases() {
    if (!someContextsNeedRendering) {
      return;
    }
    someContextsNeedRendering = false;

    // save all current WebGL state on the previous current virtual context
    if (currentVirtualContext) {
      saveAllState(currentVirtualContext._state, currentVirtualContext);
      currentVirtualContext = null;
    }

    // set the state back to the one for drawing the canvas
    restoreAllState(baseState);

    for (const vCtx of canvasToVirtualContextMap.values()) {
      if (!vCtx._needComposite) {
        continue;
      }

      vCtx._needComposite = false;
      vCtx.composite(sharedWebGLContext);
    }
  }

  window.requestAnimationFrame = (function(origFn) {
    return function(callback) {
      return origFn.call(window, (time) => {
        const result = callback(time);
        renderAllDirtyVirtualCanvases();
        return result;
      });
    };

  }(window.requestAnimationFrame));

  function setup(options) {
    Object.assign(settings, options);
  }

  window.virtualWebGL = {
    setup,
  };

}());

