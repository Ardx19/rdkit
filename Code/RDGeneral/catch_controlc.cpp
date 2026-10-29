//
//  Copyright (C) 2026 Aryan Dutt and other RDKit contributors
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//
#include <atomic>
#include <csignal>
#include <catch2/catch_all.hpp>
#include "ControlCHandler.h"
#ifdef RDK_TEST_MULTITHREADED
#include <thread>
#include <vector>
#endif

using namespace RDKit;

namespace {
using SignalHandlerType = void (*)(int);

std::atomic<int> numTestSignals{0};
void testHandler(int) {
  ++numTestSignals;
  // some platforms reset the handler before calling it
  std::signal(SIGINT, testHandler);
}

SignalHandlerType currentHandler() {
  auto current = std::signal(SIGINT, SIG_IGN);
  std::signal(SIGINT, current);
  return current;
}

// installs testHandler for one test and puts the previous handler back
class TestHandlerGuard {
 public:
  TestHandlerGuard() : d_prev(std::signal(SIGINT, testHandler)) {
    numTestSignals = 0;
  }
  ~TestHandlerGuard() { std::signal(SIGINT, d_prev); }

 private:
  SignalHandlerType d_prev;
};
}  // namespace

TEST_CASE("ControlCHandler installs and restores the SIGINT handler") {
  TestHandlerGuard guard;
  {
    ControlCHandler hdlr;
    CHECK(currentHandler() == ControlCHandler::signalHandler);
  }
  CHECK(currentHandler() == testHandler);
}

TEST_CASE("nested ControlCHandlers restore the original SIGINT handler") {
  TestHandlerGuard guard;
  {
    ControlCHandler outer;
    {
      ControlCHandler inner;
    }
    CHECK(currentHandler() == ControlCHandler::signalHandler);
  }
  CHECK(currentHandler() == testHandler);
}

TEST_CASE("the flag is kept until the next calculation starts") {
  TestHandlerGuard guard;
  {
    ControlCHandler hdlr;
    CHECK(!ControlCHandler::getGotSignal());
    std::raise(SIGINT);
    CHECK(ControlCHandler::getGotSignal());
    CHECK(numTestSignals.load() == 0);
    // a second SIGINT goes to the previous handler
    std::raise(SIGINT);
    CHECK(numTestSignals.load() == 1);
  }
  // the caller can still see that the calculation was interrupted
  CHECK(ControlCHandler::getGotSignal());
  // a SIGINT outside a calculation goes to the original handler
  std::raise(SIGINT);
  CHECK(numTestSignals.load() == 2);
  {
    ControlCHandler hdlr;
    CHECK(!ControlCHandler::getGotSignal());
  }
}

#ifdef RDK_TEST_MULTITHREADED
TEST_CASE("ControlCHandlers on several threads restore the original handler") {
  TestHandlerGuard guard;
  std::vector<std::thread> threads;
  for (unsigned int i = 0; i < 4; ++i) {
    threads.emplace_back([]() {
      for (unsigned int j = 0; j < 10000; ++j) {
        ControlCHandler hdlr;
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  CHECK(currentHandler() == testHandler);
}
#endif
