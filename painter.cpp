/*
    A TensorFlow-based 2D Cardiac Electrophysiology Modeler
    @2017 Shahriar Iravanian (siravan@emory.edu)
*/

/*
  compile as
  g++ -o libpainter.so painter.cpp screen.cpp -fPIC -shared -std=c++11 -lSDL2

*/

#include "screen.h"

extern "C" {

void *create_painter(int width, int height) {
  auto sc = new Screen(width, height);
  return sc;
}

void destroy_painter(void *screen) {
  auto sc = static_cast<Screen *>(screen);
  delete sc;
}

void imshow(void *screen, void *pixels, int width, int height) {
  auto sc = static_cast<Screen *>(screen);
  sc->callback(pixels, width, height);
}

void wait(void *screen) {
  auto sc = static_cast<Screen *>(screen);
  while(true) {
    if(sc->peek()) return;
  }
}

}
