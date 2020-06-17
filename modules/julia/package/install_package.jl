# Not using Artifacts for now
# This is a simple script to finally register the OpenCV package
# with the local package manager.

using Pkg

print(ARGS)
if size(ARGS) == 2
    Pkg.activate(ARGS[2])
end

Pkg.develop(PackageSpec(path=ARGS[1]))
