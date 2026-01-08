# ========================= matlab =========================
if(WITH_MATLAB OR MATLAB_FOUND)
  status("")
  status("  Matlab:" MATLAB_FOUND THEN "YES" ELSE "NO")
  if(MATLAB_FOUND)
    status("    mex:"         MATLAB_MEX_SCRIPT  THEN  "${MATLAB_MEX_SCRIPT}"   ELSE NO)
    status("    Compiler/generator:" MEX_WORKS   THEN  "Working"                ELSE "Not working (bindings will not be generated)")
  endif()
endif()
