function modelConvert(model, outname)
%% script for converting Piotr's matlab model into YAML format

outfile = fopen(outname, 'w');

fprintf(outfile, '%%YAML:1.0\n\n');

fprintf(outfile, ['options:\n'...
                  '    numberOfTrees: 8\n'...
                  '    numberOfTreesToEvaluate: 4\n'...
                  '    selfsimilarityGridSize: 5\n'...
                  '    stride: 2\n'...
                  '    shrinkNumber: 2\n'...
                  '    patchSize: 32\n'...
                  '    patchInnerSize: 16\n'...
                  '    numberOfGradientOrientations: 4\n'...
                  '    gradientSmoothingRadius: 0\n'...
                  '    regFeatureSmoothingRadius: 2\n'...
                  '    ssFeatureSmoothingRadius: 8\n'...
                  '    gradientNormalizationRadius: 4\n\n']);

fprintf(outfile, 'childs:\n');
printToYML(outfile, model.child', 0);

fprintf(outfile, 'featureIds:\n');
printToYML(outfile, model.fids', 0);

fprintf(outfile, 'thresholds:\n');
printToYML(outfile, model.thrs', 0);

N = 1000;
fprintf(outfile, 'edgeBoundaries:\n');
printToYML(outfile, model.eBnds, N);

fprintf(outfile, 'edgeBins:\n');
printToYML(outfile, model.eBins, N);

fclose(outfile);
gzip(outname);

end

function printToYML(outfile, A, N)
%% append matrix A to outfile as
%%    - [a11, a12, a13, a14, ..., a1n]
%%    - [a21, a22, a23, a24, ..., a2n]
%%    ...
%%
%% if size(A, 2) == 1, A is printed by N elemnent per row

    if (length(size(A)) ~= 2)
        error('printToYML: second-argument matrix should have two dimensions');
    end

    if (size(A,2) ~= 1)
        for i=1:size(A,1)
            fprintf(outfile, '    - [');
            fprintf(outfile, '%d,', A(i, 1:end-1));
            fprintf(outfile, '%d]\n', A(i, end));
        end
    else
        len = length(A);
        for i=1:ceil(len/N)
            first = (i-1)*N + 1;
             last = min(i*N, len) - 1;

            fprintf(outfile, '    - [');
            fprintf(outfile, '%d,', A(first:last));
            fprintf(outfile, '%d]\n', A(last + 1));
        end
    end
    fprintf(outfile, '\n');
end