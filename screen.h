/***********************************************************/
//  screen.h is part of the Fibulator 3D atrial fibrillation
//  simulation software
//  copyright Shahriar Iravanian (siravan@emory.edu) 2016
//
//  screen.cpp/h handles SDL calls and visualization
//
//**********************************************************/

#pragma once

#include <SDL2/SDL.h>

class Screen {
 public:
  Screen(int height, int width);
  ~Screen();

  bool peek() const;
  int getWidth() const { return width; }
  int getHeight() const { return height; }
  bool callback(void *pixels, int w, int h);

 private:
  int width;
  int height;
  SDL_Event event;
  SDL_Window *window;
  SDL_Renderer *renderer;
  SDL_Texture *texture;

  static bool quit;
};
