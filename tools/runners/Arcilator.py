#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026
#
# Use of this source code is governed by a ISC-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/ISC
#
# SPDX-License-Identifier: ISC

import os
import shlex
import shutil
import subprocess
from urllib.parse import urlparse

from BaseRunner import BaseRunner


class Arcilator(BaseRunner):
    def __init__(self):
        super().__init__(
            "Arcilator",
            executable="arcilator",
            supported_features={
                "preprocessing", "parsing", "elaboration", "simulation",
                "simulation_without_run"
            })

        self.submodule = "third_party/tools/circt-verilog"
        self.url = self._get_circt_url()

    def _get_circt_url(self):
        remote = os.environ.get("CIRCT_REMOTE",
                                "https://github.com/llvm/circt.git")
        ref = os.environ.get("CIRCT_REF") or self.get_commit()
        parsed = urlparse(remote)
        if parsed.netloc == "github.com":
            path = parsed.path[:-4] if parsed.path.endswith(".git") else parsed.path
            return f"https://github.com{path}/tree/{ref}"
        return remote

    def _tool(self, name):
        bin_dir = os.environ.get("CIRCT_BIN_DIR", "")
        if bin_dir:
            candidate = os.path.join(bin_dir, name)
            if os.path.exists(candidate):
                return candidate
        return shutil.which(name) or name

    def can_run(self):
        return all(
            os.path.exists(self._tool(tool)) or shutil.which(tool)
            for tool in ["circt-verilog", "arcilator"])

    def get_version(self):
        outputs = []
        for tool in ["circt-verilog", "arcilator"]:
            try:
                proc = subprocess.Popen(
                    [self._tool(tool), "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT)
                log, _ = proc.communicate(timeout=30)
                if proc.returncode == 0:
                    outputs.append(f"$ {tool} --version\n" +
                                   log.decode("utf-8", "ignore").strip())
                else:
                    outputs.append(f"$ {tool} --version\n{tool}")
            except (OSError, subprocess.TimeoutExpired):
                outputs.append(f"$ {tool} --version\n{tool}")
        return "\n\n".join(outputs) + "\n"

    def prepare_run_cb(self, tmp_dir, params):
        mode = params["mode"]
        scr = os.path.join(tmp_dir, "scr.sh")
        design_mlir = os.path.join(tmp_dir, "design.mlir")

        circt_verilog = self._tool("circt-verilog")
        arcilator = self._tool("arcilator")

        frontend_cmd = [circt_verilog]
        if mode == "preprocessing":
            frontend_cmd += ["-E"]
        elif mode == "parsing":
            frontend_cmd += ["--parse-only"]
        else:
            frontend_cmd += ["-o", design_mlir]

        for incdir in params["incdirs"]:
            frontend_cmd.extend(["-I", incdir])

        defines = list(params["defines"])
        if "UVM_NO_DPI" not in defines:
            defines.append("UVM_NO_DPI")

        for define in defines:
            frontend_cmd.extend(["-D", define])

        frontend_cmd += ["--timescale=1ns/1ns", "--single-unit"]
        frontend_cmd += ["-Wno-implicit-conv"]
        frontend_cmd += [
            "-Wno-error=index-oob",
            "-Wno-error=range-oob",
            "-Wno-error=range-width-oob",
        ]

        top = self.get_top_module_or_guess(params)
        if top is not None:
            frontend_cmd += ["--top=" + top]

        tags = params["tags"]
        if "ariane" in tags or "ibex" in tags:
            frontend_cmd += ["-Wno-duplicate-definition"]
        if "ariane" in tags:
            frontend_cmd += ["-Xslang=--allow-self-determined-stream-concat"]
        if "black-parrot" in tags and mode != "parsing":
            frontend_cmd += ["--allow-use-before-declare"]
            name = params["name"]
            if "bp_lce" in name or "bp_uce" in name or "bp_multicore" in name:
                frontend_cmd += ["--parse-only"]
        if "fx68k" in tags:
            frontend_cmd += ["--allow-dup-initial-drivers"]

        frontend_cmd += params["files"]

        backend_cmd = [arcilator, design_mlir, "--disable-output"]
        if mode == "simulation":
            backend_cmd = [arcilator, design_mlir, "--run"]

        self.cmd = ["sh", "scr.sh"]
        with open(scr, "w") as f:
            f.write("set -x\n")
            f.write("echo '[Arcilator] CIRCT frontend stage'\n")
            f.write(shlex.join(frontend_cmd) + " || exit $?\n")
            if mode not in {"preprocessing", "parsing"}:
                f.write("echo '[Arcilator] Arcilator backend stage'\n")
                f.write(shlex.join(backend_cmd) + " || exit $?\n")
