//
// Copyright (C) David Cosgrove 2025.
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//

#ifndef CONTROLCHANDLER_H
#define CONTROLCHANDLER_H

#include <RDGeneral/export.h>
#include <atomic>
#include <csignal>
#include <stdexcept>

namespace RDKit {
namespace detail {
// set by ControlCHandler::signalHandler(); use ControlCHandler::getGotSignal()
RDKIT_RDGENERAL_EXPORT extern std::atomic<bool> controlCGotSignal;
}  // namespace detail

class ControlCCaught : public std::runtime_error {
 public:
  explicit ControlCCaught()
      : std::runtime_error("The process was interrupted with Ctrl+c") {};
};

//! This class catches a control-C/SIGINT and sets a flag (see getGotSignal())
//! if one is received.  It is intended to be used inside a long
//! C++ calculation called from Python which intercepts the signal
//! handler.  The C++ code must check the value of getGotSignal()
//! periodically and act accordingly.
//! The signal handler is installed when the first ControlCHandler is
//! created and the previous handler is put back when the last one is
//! destroyed, so ControlCHandlers can be nested and can exist on several
//! threads at the same time.  The flag is cleared when the first one is
//! created and kept after the last one is destroyed, so it can still be
//! checked once the calculation has returned.
//! Example usage, inside a boost::python wrapper:
//!  ResultsObject results;
//!  {
//!   NOGIL gil;
//!    results = someFunction();
//!  }
//!  if (results.getCancelled()) {
//!    throw_runtime_error("someFunction cancelled");
//!  }
//! It's important that the exception is thrown once the GIL has been
//! released, otherwise a crash is inevitable at some future point.
class RDKIT_RDGENERAL_EXPORT ControlCHandler {
 public:
  ControlCHandler();
  ControlCHandler(const ControlCHandler &) = delete;
  ControlCHandler(ControlCHandler &&) = delete;
  ControlCHandler &operator=(const ControlCHandler &) = delete;
  ControlCHandler &operator=(ControlCHandler &&) = delete;
  ~ControlCHandler();
  // inline, since substructure searches check it in their inner loop
  static bool getGotSignal() {
    return detail::controlCGotSignal.load(std::memory_order_relaxed);
  }
  static void signalHandler(int signalNumber);
  //! clears the flag and, if a ControlCHandler exists, installs the
  //! signal handler again
  static void reset();
};
}  // namespace RDKit
#endif  // CONTROLCHANDLER_H
