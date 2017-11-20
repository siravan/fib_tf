/***********************************************************/
//  screen.h is part of the Fibulator 3D atrial fibrillation
//  simulation software
//  copyright Shahriar Iravanian (siravan@emory.edu) 2016
//
//  screen.cpp/h handles SDL calls and visualization
//
//**********************************************************/

#include <cassert>
#include <iostream>

#include "screen.h"

bool Screen::quit = false;

Screen::Screen(int height_, int width_) : width(width_), height(height_) {
  if (!SDL_WasInit(SDL_INIT_VIDEO | SDL_INIT_EVENTS) &&
      SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) < 0) {
    std::cerr << "SDL Init Error: " << SDL_GetError() << '\n';
    return;
  }

  window = SDL_CreateWindow("3D AF Fibulator", SDL_WINDOWPOS_UNDEFINED,
                            SDL_WINDOWPOS_UNDEFINED, width, height, 0);

  renderer = SDL_CreateRenderer(window, -1, 0);
  texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                              SDL_TEXTUREACCESS_STREAMING, width, height);
}

Screen::~Screen() {
  SDL_DestroyTexture(texture);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  /*
    TODO: SDL_QUIT is commented out due to an error in IPython when
    using matplotlib after closing SDL
    The error is "arguments to dbus_connection_unref() were incorrect,
      assertion "connection->generation == _dbus_current_generation" failed
      in file dbus-connection.c line 2824."

  */
  std::cerr << "warning: Screen closed without calling SDL_Quit\n";
  // SDL_Quit();
}

bool Screen::callback(void *pixels, int w, int h) {
  assert(w == width && h == height);
  SDL_UpdateTexture(texture, NULL, pixels, w * sizeof(uint32_t));
  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, texture, NULL, NULL);
  SDL_RenderPresent(renderer);
  SDL_PumpEvents();
  return peek();
}

bool Screen::peek() const {
  SDL_Event event;

  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_QUIT ||
        event.type == SDL_WINDOWEVENT &&
            event.window.event == SDL_WINDOWEVENT_CLOSE)
      Screen::quit = true;
  }

  return Screen::quit;  // note: quit is a static member
}
