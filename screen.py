"""
    A TensorFlow-based 2D Cardiac Electrophysiology Modeler
    @2017-2018 Shahriar Iravanian (siravan@emory.edu)
"""

# from dll import _bind
from ctypes import *
import numpy as np
#import PIL.Image
from matplotlib import colors

# in linux, we can simply use the following code instead of importing from dll
#
SDL = CDLL('libSDL2.so')
#
def _bind(fun, argtypes, restype):
    f = getattr(SDL, fun, None)
    f.argtypes = argtypes
    f.restype = restype
    return f

TTF = CDLL('libSDL2_ttf.so')

def _ttf_bind(fun, argtypes, restype):
    f = getattr(TTF, fun, None)
    if f is None:
        raise Exception('function not found in the TTF library')
    f.argtypes = argtypes
    f.restype = restype
    return f

SDL_INIT_VIDEO = 0x00000020
SDL_INIT_EVENTS = 0x00004000
SDL_WINDOWPOS_UNDEFINED = 0x1fff0000
SDL_PIXELFORMAT_ARGB8888 = 0x16362004
SDL_TEXTUREACCESS_STREAMING = 0x00000001
SDL_QUIT = 0x00000100
SDL_WINDOWEVENT = 0x00000200
SDL_WINDOWEVENT_CLOSE = 0x00000020
SDL_ALPHA_OPAQUE = 255
SDL_ALPHA_TRANSPARENT = 0
SDL_HINT_RENDER_SCALE_QUALITY = "SDL_RENDER_SCALE_QUALITY"

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
    def __init__(self, height, width, caption=None, font_size=16):
        """
            Creates a blank screen

            Args:
                height: the screen height in pixels
                width: the screen width in pixels
                caption: a string
                font_size
        """
        # first, we will declare SDL2 functions we need...

        # int SDL_Init(c_uint32 flags)
        self._Init = _bind('SDL_Init', [c_uint32], c_int)

        # void SDL_Quit(void)
        self._Quit = _bind('SDL_Quit', [], None)

        # c_uint32 SDL_WasInit(c_uint32 flags)
        self._WasInit = _bind('SDL_WasInit', [c_uint32], c_int)

        # SDL_Window* SDL_CreateWindow(const char* title, int x, int y, int w, int h, c_uint32 flags)
        self._CreateWindow = _bind('SDL_CreateWindow', [c_char_p, c_int, c_int, c_int, c_int, c_uint32], c_void_p)

        # SDL_Renderer* SDL_CreateRenderer(SDL_Window* window,  int index, c_uint32 flags)
        self._CreateRenderer = _bind('SDL_CreateRenderer', [c_void_p, c_int, c_uint32], c_void_p)

        # SDL_Texture* SDL_CreateTexture(SDL_Renderer* renderer, c_uint32 format, int access, int w, int h)
        self._CreateTexture = _bind('SDL_CreateTexture', [c_void_p, c_uint32, c_int, c_int, c_int], c_void_p)

        # SDL_Texture* SDL_CreateTextureFromSurface(SDL_Renderer* renderer, SDL_Surface*  surface)
        self._CreateTextureFromSurface = _bind('SDL_CreateTextureFromSurface', [c_void_p, c_void_p], c_void_p)

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

        # int SDL_SetRenderDrawColor(SDL_Renderer* renderer, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
        self._SetRenderDrawColor = _bind('SDL_SetRenderDrawColor', [c_void_p, c_ubyte, c_ubyte, c_ubyte, c_ubyte], None)

        # int SDL_RenderDrawLine(SDL_Renderer* renderer, int x1, int y1, int x2, int y2)
        self._RenderDrawLine = _bind('SDL_RenderDrawLine', [c_void_p, c_int, c_int, c_int, c_int], None)

        # int SDL_RenderDrawLines(SDL_Renderer* renderer, const SDL_Point* points, int count)
        # note: SDL_Point is { int x, y}
        self._RenderDrawLines = _bind('SDL_RenderDrawLines', [c_void_p, POINTER(c_int), c_int], None)

        # SDL_bool SDL_SetHint(const char* name, const char* value)
        self._SetHint = _bind('SDL_SetHint', [c_char_p, c_char_p], None)

        # void SDL_FreeSurface(SDL_Surface* surface)
        self._FreeSurface = _bind('SDL_FreeSurface', [c_void_p], None)

        # void SDL_PumpEvents(void)
        self._PumpEvents = _bind('SDL_PumpEvents', [], None)

        # int SDL_PollEvent(SDL_Event* event)
        self._PollEvent = _bind('SDL_PollEvent', [POINTER(SDL_WindowEvent)], c_int)

        # int TTF_Init()
        self._TTF_Init = _ttf_bind('TTF_Init', [], c_int)

        # int TTF_Quit()
        self._TTF_Quit = _ttf_bind('TTF_Quit', [], None)

        # TTF_Font *TTF_OpenFont(const char *file, int ptsize)
        self._OpenFont = _ttf_bind('TTF_OpenFont', [c_char_p, c_int], c_void_p)

        # SDL_Surface *TTF_RenderText_Solid(TTF_Font *font, const char *text, SDL_Color fg)
        self._RenderText_Solid = _ttf_bind('TTF_RenderText_Solid', [c_void_p, c_char_p, c_int], c_void_p)

        # void TTF_CloseFont(TTF_Font *font)
        self._CloseFont = _ttf_bind('TTF_CloseFont', [c_void_p], None)

        # int TTF_SizeText(TTF_Font *font, const char *text, int *w, int *h)
        self._SizeText = _ttf_bind('TTF_SizeText', [c_void_p, c_char_p, POINTER(c_int), POINTER(c_int)], None)

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

        self._SetHint(bytes(SDL_HINT_RENDER_SCALE_QUALITY, encoding='utf8'),
                      bytes("1", encoding='utf8'))

        self.colors = dict(colors.BASE_COLORS, **colors.CSS4_COLORS)

        self.image = None

        self._painting = False
        self.clear()

        if self._TTF_Init() != 0:
            print("cannot initialize truetype fonts")
            self._font = None
        else:
            self._font = self._OpenFont(bytes('UbuntuMono-R.ttf', encoding='utf8'), font_size)

    def destroy(self):
        """
            destroys the screen and releases the relevant resources
        """
        self._CloseFont(self._font)
        self._TTF_Quit()

        self._DestroyTexture(self._texture)
        self._DestroyRenderer(self._renderer)
        self._DestroyWindow(self._window)
        # self._Quit()

    def color(self, name):
        """
            color is a helper function and converts a color name into a color
            triplet, i.e., [r, g, b] list

            Arg:
                name: a string representing the color name (matplotlib scheme)
        """
        c = self.colors[name]
        rgb = colors.to_rgb(c)
        return (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

    def color_val(self, color):
        """
            converts a color triplet as returned by Screen.color into a
            uint32 format
        """
        return color[0] | (color[1]<<8) | (color[2]<<16)

    def begin_paint(self):
        """
            begin_paint opens a paint transactions. It clears an off screen
            buffer. All drawing operations before an end_paint are drawn in
            the buffer and all not immediately shown on the screen.
        """
        if self._painting:
            raise Exception('begin/end paint is not reentrant!')
        self._SetRenderDrawColor(self._renderer, 0, 0, 0, SDL_ALPHA_OPAQUE)
        self._RenderClear(self._renderer)
        self._painting = True

    def end_paint(self):
        """
            end_paints closes a paint transaction and paints the off screen
            buffer on the screen.
        """
        if self._painting:
            self._RenderPresent(self._renderer)
            self._painting = False
        else:
            raise Exception('end_paint called without begin_paint')
        self._PumpEvents()

    def present(self):
        """
            present is a helper function. It is called by the drawing operations
            to paint the result immediately if a transation is not open.
        """
        if not self._painting:
            self._RenderPresent(self._renderer)
            self._PumpEvents()

    def imshow(self, image):
        """
            imshow paints a numpy ndarray, which should be the same size as the Screen

            if x is a 2D array of dtype uint32 or int32, it is copied directly
            to the SDL buffer (expecting an ARGB format)

            if x is a 2D array of float type, it is assumed to be gray-scaled,
            where each pixel should be between 0 and 1.
            Note: no check is done to make sure the values are in the range.

            if x is a 3D array of float type, it is assumed to be color with
            red = x[:,:,0], green = x[:,:,1] and blue = x[:,:,2]
        """
        if image.dtype == np.uint32 or image.dtype == np.int32:
            a = image
        elif image.dtype == np.float32 or image.dtype == np.float64:
            if len(image.shape) == 2 or image.shape[2] == 1:
                a = np.uint8(np.squeeze(image) * 255) * 0x010101 + 0xff000000
            elif x.shape[2] == 3:
                a = np.c_uint32(np.uint8(image[:,:,2]*255) +
                                np.uint8(image[:,:,1]*255) * 256 +
                                np.uint8(image[:,:,0]*255) * 65536)
            else:
                raise TypeError('float ndarrays to imshow should be of form MxNx1 or MxNx3')
        else:
            raise TypeError('ndarrays to imshow should be of type unit32, int32, float32, or float64')

        self.image = a  # save the buffer to save to file

        pixels = a.ctypes.data_as(POINTER(c_float))
        self._UpdateTexture(self._texture, None, pixels, sizeof(c_uint32) * self._width)
        self._RenderClear(self._renderer)
        self._RenderCopy(self._renderer, self._texture, None, None)
        self.present()

    def clear(self):
        """
            clear clears the drawing surface and fill it with black color
        """
        self._SetRenderDrawColor(self._renderer, 0, 0, 0, SDL_ALPHA_OPAQUE)
        self._RenderClear(self._renderer)
        self.present()

    def plot(self, x, y, color):
        """
            plot plots an X/Y graph
            the coordinates are in the pixel unit

            args:
                x: an ndarray of the x-coordinate of the points
                y: am ndarray of the y-coordinate of the points
                color: a string of the color name
        """
        x = x.astype(np.int32)
        y = y.astype(np.int32)
        points = np.dstack([x,y]).ravel()
        c = self.color(color)
        self._SetRenderDrawColor(self._renderer, c[0], c[1], c[2], SDL_ALPHA_OPAQUE)
        self._RenderDrawLines(self._renderer, points.ctypes.data_as(POINTER(c_int)), x.shape[0])
        self.present()

    def draw_text(self, text, x, y, color):
        """
            draw_text draws a string on the screen using the font_size
            sent to the Screen.__init__.

            Args:
                text: the string to draw
                x: the x-coordiante
                y: the y-coordiante
                color: a color name
        """
        if self._font is None:
            print('truetype fonts are not available')
            return
        s = bytes(text, encoding='utf8')
        w = c_int()
        h = c_int()
        self._SizeText(self._font, s, byref(w), byref(h))
        surf = self._RenderText_Solid(self._font, s, self.color_val(self.color(color)))
        texture = self._CreateTextureFromSurface(self._renderer, surf)
        rect = np.array([x, y, w.value, h.value], dtype=np.int32)
        self._RenderCopy(self._renderer, texture, None, rect.ctypes.data_as(POINTER(c_int)))
        self._FreeSurface(surf)
        self._DestroyTexture(texture)
        self.present()

    def peek(self):
        """
            Checks SDL events and returns True if a Quit event is detected
        """
        event = SDL_WindowEvent()

        while not self._quit and self._PollEvent(event) == 1:
            if (event.type == SDL_QUIT or
                (event.type == SDL_WINDOWEVENT and
                event.event == SDL_WINDOWEVENT_CLOSE)):
                self._quit = True
        return self._quit

    def wait(self):
        """
            Waits until a Quit event is detected, then closes the Screen
        """
        while not self.peek():
            pass
        self.destroy()

    def save(self, name):
        """
            saves the screen as a PNG fine

            Args:
                name: PNG file name
        """
        if self.image is not None:
            PIL.Image.fromarray(self.image, mode='RGBA').save(name, 'png')
        else:
            print('No suitable image to save!')
