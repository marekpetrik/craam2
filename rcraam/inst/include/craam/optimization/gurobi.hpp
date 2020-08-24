// This file is part of CRAAM, a C++ library for solving plain
// and robust Markov decision processes.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "craam/definitions.hpp"
#include <memory>

namespace craam {

#ifdef GUROBI_USE

#include <gurobi_c++.h>

/**
 * Constructs a static instance of the gurobi object.
 * The returned object should not be used concurrently,
 * but is definitely being used that way!
 *
 * The construction is also not thread-safe.
 */
inline std::shared_ptr<GRBEnv> get_gurobi() {
    static std::shared_ptr<GRBEnv> env;
    if (env == nullptr) {
        try {
            env = std::make_shared<GRBEnv>();
        } catch (std::exception& e) {
            std::cerr << "Problem constructing Gurobi object: " << std::endl
                      << e.what() << std::endl;
            throw e;
        } catch (...) {
            std::cerr << "Unknown exception while creating a gurobi object. Could be a "
                         "license problem."
                      << std::endl;
            throw;
        }
        env->set(GRB_IntParam_OutputFlag, 0);
    }
    return env;
}
#endif

} // namespace craam
