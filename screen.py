"""
    A TensorFlow-based 2D Cardiac Electrophysiology Modeler
    @2017 Shahriar Iravanian (siravan@emory.edu)
"""

from dll import _bind
from ctypes import *
import numpy as np

# in linux, we can simply use the following code instead of importing from dll
#
#SDL = CDLL('libSDL2.so')
#
#def _bind(fun, argtypes, restype):
#    f = getattr(SDL, fun, None)
#    f.argtypes = argtypes
#    f.restype = restype
#    return f

SDL_INIT_VIDEO = 0x00000020
SDL_INIT_EVENTS = 0x00004000
SDL_WINDOWPOS_UNDEFINED = 0x1fff0000
SDL_PIXELFORMAT_ARGB8888 = 0x16362004
SDL_TEXTUREACCESS_STREAMING = 0x00000001
SDL_QUIT = 0x00000100
SDL_WINDOWEVENT = 0x00000200
SDL_WINDOWEVENT_CLOSE = 0x00000020


class SDL_WindowEvent(Structure):
    _fields_ = [("type", c_uint32),
                ("timestamp", c_uint32),
                ("windowID", c_uint32),
                ("event", c_uint8),
                ("padding1", c_uint8),
                ("padding2", c_uint8),
                ("padding3", c_uint8),
                ("data1", c_int32),
                ("data2", c_int32),
                ("padding", c_uint8 * 56)
                ]


class Screen:
    """
        A window plotting 2D numpy ndarrays
    """
    def __init__(self, height, width, caption=None):
        """
            Creates a blank screen
        """
        # first, we will declare SDL2 functions we need...

        # int SDL_Init(c_uint32 flags)
        self._Init = _bind('SDL_Init', [c_uint32], c_int)

        # c_uint32 SDL_WasInit(c_uint32 flags)
        self._WasInit = _bind('SDL_WasInit', [c_uint32], c_int)

        # SDL_Window* SDL_CreateWindow(const char* title, int x, int y, int w, int h, c_uint32 flags)
        self._CreateWindow = _bind('SDL_CreateWindow', [c_char_p, c_int, c_int, c_int, c_int, c_uint32], c_void_p)

        # SDL_Renderer* SDL_CreateRenderer(SDL_Window* window,  int index, c_uint32 flags)
        self._CreateRenderer = _bind('SDL_CreateRenderer', [c_void_p, c_int, c_uint32], c_void_p)

        # SDL_Texture* SDL_CreateTexture(SDL_Renderer* renderer, c_uint32 format, int access, int w, int h)
        self._CreateTexture = _bind('SDL_CreateTexture', [c_void_p, c_uint32, c_int, c_int, c_int], c_void_p)

        # void SDL_DestroyTexture(SDL_Texture* texture)
        self._DestroyTexture = _bind('SDL_DestroyTexture', [c_void_p], None)

        # void SDL_DestroyRenderer(SDL_Renderer* renderer)
        self._DestroyRenderer = _bind('SDL_DestroyRenderer', [c_void_p], None)

        # void SDL_DestroyWindow(SDL_Window* window)
        self._DestroyWindow = _bind('SDL_DestroyWindow', [c_void_p], None)

        # int SDL_UpdateTexture(SDL_Texture* texture, const SDL_Rect* rect, const void *pixels, int pitch)
        self._UpdateTexture = _bind('SDL_UpdateTexture', [c_void_p, c_void_p, c_void_p, c_int], c_int)

        # int SDL_RenderClear(SDL_Renderer* renderer)
        self._RenderClear = _bind('SDL_RenderClear', [c_void_p], None)

        # int SDL_RenderCopy(SDL_Renderer *renderer, SDL_Texture *texture, const SDL_Rect* srcrect, const SDL_Rect* dstrect)
        self._RenderCopy = _bind('SDL_RenderCopy', [c_void_p, c_void_p, c_void_p, c_void_p], None)

        # void SDL_RenderPresent(SDL_Renderer* renderer)
        self._RenderPresent = _bind('SDL_RenderPresent', [c_void_p], None)

        # void SDL_PumpEvents(void)
        self._PumpEvents = _bind('SDL_PumpEvents', [], None)

        # int SDL_PollEvent(SDL_Event* event)
        self._PollEvent = _bind('SDL_PollEvent', [POINTER(SDL_WindowEvent)], None)

        self._quit = False

        if (self._WasInit(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0 and
            self._Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) < 0):
            print('Error in initializing SDL')
            return

        if caption is None:
            caption = 'SDL Window'

        self._window = self._CreateWindow(bytes(caption, encoding='utf8'),
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, 0)
        self._width = width
        self._height = height

        self._renderer = self._CreateRenderer(self._window, -1, 0)
        self._texture = self._CreateTexture(self._renderer,
            SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, width, height)

    def destroy(self):
        self._DestroyTexture(self._texture)
        self._DestroyRenderer(self._renderer)
        self._DestroyWindow(self._window)

    def imshow(self, x):
        """
            Paints a numpy ndarray, which should be the same size as the Screen

            if x is a 2D array of dtype uint32 or int32, it is copied directly
            to the SDL buffer (expecting an ARGB format)

            if x is a 2D array of float type, it is assumed to be gray-scaled,
            where each pixel should be between 0 and 1.
            Note: no check is done to make sure the values are in the range.

            if x is a 3D array of float type, it is assumed to be color with
            red = x[:,:,0], green = x[:,:,1] and blue = x[:,:,2]
        """
        assert(x.shape[0] == self._height and x.shape[1] == self._width)

        if x.dtype == np.uint32 or x.dtype == np.int32:
            a = x
        elif x.dtype == np.float32 or x.dtype == np.float64:
            if len(x.shape) == 2 or x.shape[2] == 1:
                a = np.uint8(np.squeeze(x) * 255) * 0x010101
            elif x.shape[2] == 3:
                a = np.c_uint32(np.uint8(x[:,:,2]*255) +
                                np.uint8(x[:,:,1]*255) * 256 +
                                np.uint8(x[:,:,0]*255) * 65536)
            else:
                raise TypeError('float ndarrays to imshow should be of form MxNx1 or MxNx3')
        else:
            raise TypeError('ndarrays to imshow should be of type unit32, int32, float32, or float64')

        pixels = a.ctypes.data_as(POINTER(c_float))
        self._UpdateTexture(self._texture, None, pixels, sizeof(c_uint32) * self._width)
        self._RenderClear(self._renderer)
        self._RenderCopy(self._renderer, self._texture, None, None)
        self._RenderPresent(self._renderer)
        self._PumpEvents()

    def peek(self):
        """
            Checks SDL events and returns True if a Quit event is detected
        """
        event = SDL_WindowEvent()

        while self._PollEvent(event) != 1:
            if (event.type == SDL_QUIT or
                (event.type == SDL_WINDOWEVENT and
                event.event == SDL_WINDOWEVENT_CLOSE)):
                self._quit = True
                break
        return self._quit

    def wait(self):
        """
            Waits until a Quit event is detected, then closes the Screen
        """
        while not self.peek():
            pass
        self.destroy()
