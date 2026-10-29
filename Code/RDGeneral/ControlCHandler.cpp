//
// Copyright (C) David Cosgrove 2025.
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//

#include "ControlCHandler.h"

#include <atomic>
#ifdef RDK_BUILD_THREADSAFE_SSS
#include <mutex>
#endif

namespace RDKit {
namespace detail {
// used in the signal handler, so it must be a lock-free atomic
std::atomic<bool> controlCGotSignal{false};
}  // namespace detail

namespace {
using SignalHandlerType = void (*)(int);
// read in the signal handler, so it must be a lock-free atomic
std::atomic<SignalHandlerType> prevHandler{SIG_DFL};
// The number of ControlCHandlers that exist, on any thread.  The state is
// kept here, in one library, so that every library sees the same flag.
// Changes between 0 and 1, which install or restore the signal handler, are
// made while holding handlerMutex; the others don't need it.
std::atomic<unsigned int> numHandlers{0};
#ifdef RDK_BUILD_THREADSAFE_SSS
std::mutex handlerMutex;
#endif
}  // namespace

ControlCHandler::ControlCHandler() {
  // if the handler is installed already, just count this one
  auto n = numHandlers.load();
  while (n > 0) {
    if (numHandlers.compare_exchange_weak(n, n + 1)) {
      return;
    }
  }
#ifdef RDK_BUILD_THREADSAFE_SSS
  std::lock_guard<std::mutex> lock(handlerMutex);
#endif
  if (numHandlers++ == 0) {
    detail::controlCGotSignal = false;
    prevHandler = std::signal(SIGINT, signalHandler);
  }
}

ControlCHandler::~ControlCHandler() {
  // if other ControlCHandlers remain, just uncount this one
  auto n = numHandlers.load();
  while (n > 1) {
    if (numHandlers.compare_exchange_weak(n, n - 1)) {
      return;
    }
  }
#ifdef RDK_BUILD_THREADSAFE_SSS
  std::lock_guard<std::mutex> lock(handlerMutex);
#endif
  if (--numHandlers == 0) {
    if (const auto prev = prevHandler.load(); prev != SIG_ERR) {
      std::signal(SIGINT, prev);
    }
  }
}

void ControlCHandler::signalHandler(int signalNumber) {
  if (signalNumber == SIGINT) {
    detail::controlCGotSignal = true;
    // a second SIGINT goes to the previous handler
    std::signal(SIGINT, prevHandler.load());
  }
}

void ControlCHandler::reset() {
#ifdef RDK_BUILD_THREADSAFE_SSS
  std::lock_guard<std::mutex> lock(handlerMutex);
#endif
  detail::controlCGotSignal = false;
  if (numHandlers > 0) {
    std::signal(SIGINT, signalHandler);
  }
}
}  // namespace RDKit
