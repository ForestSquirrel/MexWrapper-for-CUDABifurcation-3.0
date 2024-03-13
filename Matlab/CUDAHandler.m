classdef CUDAHandler
%% Bifurcation3.0 MATLAB Interface
% Provides easy to use way to run CUDA computations directly from MATLAB
% CUDAHandler will handle everything you need from compilation of MEX API CUDA interfaces
% to validating inputs. All function support ("Name",Value) syntax or
% (Name=Value) which is the best way to always keep up with your research.
% CUDAHandler properties provide an easy way to run different types of
% analysis with same mutual inputs (e.g. with similar tMax - Simulation time).
% <Compiled> folder is used to store all compiled MEX API files which
% provides an easy way to utilize your dinamical systems later. In
% <Compiled> folder a new folder for every system is created upon
% constructing CUDAHandler object, the folder willl always be named as an
% input string passed to the constructor. If the folder already exists
% instead of compiling CUDAHandler will use precompiled MEXs from there.
% Usage:
%   <class obj> = CUDAHandler("<System Name>");
% Accesing global parameters:
%   <class obj>.gOpts();
%   Will print to console all global parameters with descriptions and
%   expexted values, just copy-paste to your source file
% Supported types of analysis:
%   1D bifurcation (for any system parameter/initial conditions/integration step)
%       [xData, yData] = <class obj>.bifurcation1D(args);
%       [xData, yData] = <class obj>.bifurcation1DIC(args);
%       [xData, yData] = <class obj>.bifurcation1DForH(args);
%   2D bifurcation (for any pair of system parameters/initial conditions)
%       [xData, yData, cData] = <class obj>.bifurcation2D(args);
%       [xData, yData, cData] = <class obj>.bifurcation2DIC(args);
%   1D Lyapunov exponents (for any system parameter/initial conditions)
%       [xData, yData] = <class obj>.LLE1D(args);
%       [xData, yData] = <class obj>.LLE1DIC(args);
%   2D Lyapunov exponents (for any pair of system parameters/initial conditions)
%       [xData, yData, cData] = <class obj>.LLE2D(args);
%       [xData, yData, cData] = <class obj>.LLE2DIC(args);
%   1D LS metrics (for any system parameter)
%       [xData, yData] = <class obj>.LS1D(args);
%   2D LS metrics (for any pair of system parameters)
%       [xData, yData, cData] = <class obj>.LS2D(args);
%
% Documentation is available from MATLAB directly
%   help CUDAHandler
%   help CUDAHandler.<fcn>
%   You can alway click hyperlink which appears in console after running
%   help CUDAHandler for details
    properties (Access = private)
        SystemName string
        LibPath string = 'Library\'
        MEXPath string = 'MEXWrappers\'
        wrappers = {'bifurcation1DForHMEXWrapper', ...
                        'bifurcation1DICMEXWrapper', ...
                        'bifurcation1DMEXWrapper', ...
                        'bifurcation2DICMexWrapper', ...
                        'bifurcation2DMexWrapper', ...
                        'LLE1DICMEXWrapper', ...
                        'LLE1DMEXWrapper', ...
                        'LLE2DICMEXWrapper', ...
                        'LLE2DMEXWrapper', ...
                        'LS1DMEXWrapper', ...
                        'LS2DMEXWrapper'};
    end

    % Public properties
    % Used as global arguments to differnet types of analysis
    % While using them is optional they provide a great way to
    % simplify running different types of analysis with same inputs
    properties (Access = public)
        tMax (1,1) double {mustBePositive(tMax)} = 500                                   % Simultaion time
        NT (1, 1) double {mustBePositive(NT)} = 0.4                                      % Normaliztion time for LLE
        nPts (1, 1) double {mustBePositive(nPts)} = 100                                  % Number of point to analyze system in
        h (1, 1) double {mustBePositive(h)} = 0.01                                       % Integration step
        LLE_eps (1, 1) double {mustBePositive(LLE_eps)} = 1e-3                           % Epsilon value for LLE
        intCon (1, :) double {mustBeNumeric(intCon)} = [0, 1, 0]                         % Initial conditions
        ranges (:, 2) double {mustBeNumeric(ranges)} = [0, 10; 0, 10]                    % Ranges to vary mutVariables
        indicesOfMutVars (1, :) double {mustBeNonnegative(indicesOfMutVars)} = [3, 4]    % Indices of mutVariabes
        writableVar (1, 1) double {mustBeNonnegative(writableVar)} = 0                   % Indice of state variable to conduct analysys on
        maxValue (1, 1) double {mustBeNumeric(maxValue)} = 1000                          % Value to determine if system has diverged
        transient (1, 1) double {mustBeNonnegative(transient)} = 500                     % Time to simulate transient
        values (1, :) double {mustBeNumeric(values)} = [0.02, -1, 1, 3, 1]               % System parameters
        preScaller (1, 1) double {mustBeNonnegative(preScaller)} = 1                     % Modifier to reduce computations (every <preScaller> point is computed)
        DBSCAN_eps (1, 1) double {mustBeNonnegative(DBSCAN_eps)} = 0.01                  % Epsilon value for DBSCAN
        LS_eps (1, 1) double {mustBeNonnegative(LS_eps)} = 1e-5                          % Epsilon value for LS
    end

    methods
        function obj = CUDAHandler(SystemName)
            %% Constructor for CUDAHandler
            % Uses <SystemName> string as an input argument
            % Checks <../Compiled/SystemName> folder, if one exists uses it
            % to access all precompiled MEXs there
            % If folder with specified <SystemName> does not exist, creates
            % folder and compiles MEXs for all CUDA lib functions and puts
            % them in created folder. The folder than added to MATLAB
            % paths.
            obj.SystemName = SystemName;
            compiledDir = fullfile('Compiled', SystemName);
            if ~exist(compiledDir, 'dir')
                fprintf('Specified system unrecognized \n Starting compilation... \n');
                mkdir(compiledDir);
                obj.compileWrappers(compiledDir);
            end
            addpath(compiledDir);
            fprintf(['CUDAHandler for system %s ready! \n' ...
                     'Consider setting global args for CUDA to simplify analysis \n'], obj.SystemName);
        end
    end

    methods (Access = private)
        function compileWrappers(obj, compiledDir)
            %% Method to compile all MEXs
            % Only used in class constructor
            % Uses dynamic naming based on project structure and
            % <SystemName> to compile MEXs with NVCC and move them to
            % specified directory
            for i = 1:length(obj.wrappers)
                wrapperName = obj.wrappers{i};
                fprintf('Compiling %s \n', obj.wrappers{i});
                mexcudaCommand = sprintf('mexcuda -output %s_%s %s%s.cpp %scudaLibrary.cu %scudaMacros.cu %shostLibrary.cu', ...
                                        obj.SystemName, wrapperName, obj.MEXPath, wrapperName, obj.LibPath, obj.LibPath, obj.LibPath);
                eval(mexcudaCommand);
                movefile(sprintf('%s_%s.mexw64', obj.SystemName, wrapperName), compiledDir);
            end
            fprintf('All wrappers are compiled! \n');
        end
    end

    methods (Access = public)
        % Methods to run CUDA analysis
        function [xData, yData, cData] = LLE2D(obj, args)
        %% LLE2D
        % Computes 2 dimensional Lyapunov exponent diagram
        % for varying parameters of system. It is much like 2D
        % bifurcation diagram, but bifurcation criteria is Lyapunov exponent
        % Inputs:
        %   tMax (1, 1) double -  Simultaion time
        %   NT (1, 1) double - Normaliztion time for LLE
        %   nPts (1, 1) double - Number of point to analyze system in
        %   h (1, 1) double - Integration step
        %   LLE_eps (1, 1) double - Epsilon value for LLE
        %   intCon (1, :) double - Initial conditions
        %   ranges (2, 2) double - Ranges to vary mutVariables
        %   indicesOfMutVars (1, 2) double - Indices of mutVariabes
        %   writableVar (1, 1) double - Indice of state variable to conduct analysis on
        %   maxValue (1, 1) double - Value to determine if system has diverged
        %   transient (1, 1) double - Time to simulate transient
        %   values (1, :) double - System parameters
        % Outputs:
        %   xData (1, N) double - Array of values for 1st mutVar
        %   yData (1, N) double - Array of values for 2nd mutVar
        %   cData (N, N) double - Matrix of Lyapunov exponent values
            arguments( Input )
                obj (1, 1) CUDAHandler
                args.tMax (1, 1) double {mustBePositive(args.tMax)} = obj.tMax                                           % Simultaion time
                args.NT (1, :) double {mustBePositive(args.NT)} = obj.NT                                                 % Normaliztion time for LLE
                args.nPts (1, 1) double {mustBePositive(args.nPts)} = obj.nPts                                           % Number of point to analyze system in
                args.h (1, 1) double {mustBePositive(args.h)} = obj.h                                                    % Integration step
                args.LLE_eps (1, 1) double {mustBePositive(args.LLE_eps)} = obj.LLE_eps                                  % Epsilon value for LLE
                args.intCon (1, :) double {mustBeNumeric(args.intCon)} = obj.intCon                                      % Initial conditions
                args.ranges (2, 2) double {mustBeNumeric(args.ranges)} = obj.ranges                                      % Ranges to vary mutVariables
                args.indicesOfMutVars (1, 2) double {mustBeNonnegative(args.indicesOfMutVars)} = obj.indicesOfMutVars    % Indices of mutVariabes
                args.writableVar (1, 1) double {mustBeNonnegative(args.writableVar)} = obj.writableVar                   % Indice of state variable to conduct analysys on
                args.maxValue (1, 1) double {mustBeNumeric(args.maxValue)} = obj.maxValue                                % Value to determine if system has diverged
                args.transient (1, 1) double {mustBeNonnegative(args.transient)} = obj.transient                         % Time to simulate transient
                args.values (1, :) double {mustBeNumeric(args.values)} = obj.values                                      % System parameters
            end
            arguments( Output )
                xData
                yData
                cData
            end
            % Reshape ranges to CUDA style
            args.ranges = reshape(args.ranges', 1, []);

            % Call CUDA
            [xData, yData, cData] = feval(sprintf('%s_LLE2DMEXWrapper', obj.SystemName), ...
                args.tMax, ...
                args.NT, ...
                args.nPts, ...
                args.h, ...
                args.LLE_eps, ...
                args.intCon, ...
                numel(args.intCon), ...
                args.ranges, ...
                args.indicesOfMutVars, ...
                args.writableVar, ...
                args.maxValue, ...
                args.transient, ...
                args.values, ...
                numel(args.values));
        end
        
        function [xData, yData] = bifurcation1DForH(obj, args)
        %% bifurcation1DForH
        % Computes 1 dimensional bifurcation diagram for integration step
        % Inputs: 
        %   tMax (1, 1) double - Simultaion time
        %   nPts (1, 1) double - Number of point to analyze system in
        %   intCon (1, :) double - Initial conditions
        %   ranges (1, 2) double - Ranges to vary h
        %   writableVar (1, 1) double - Indice of state variable to conduct analysys on
        %   maxValue (1, 1) double - Value to determine if system has diverged
        %   transient (1, 1) double -Time to simulate transient
        %   values (1, :) double - System parameters
        %   preScaller (1, 1) double - Modifier to reduce computations (every <preScaller> point is computed)
        % Outputs:
        %   xData (1, N) double - Array of values for 1st mutVar
        %   yData (1, N) double - Array of amplitude peaks for writableVar 
            arguments( Input )
                obj (1, 1) CUDAHandler
                args.tMax (1, 1) double {mustBePositive(args.tMax)} = obj.tMax                                           % Simultaion time
                args.nPts (1, 1) double {mustBePositive(args.nPts)} = obj.nPts                                           % Number of point to analyze system in
                args.intCon (1, :) double {mustBeNumeric(args.intCon)} = obj.intCon                                      % Initial conditions
                args.ranges (1, 2) double {mustBeNumeric(args.ranges)} = obj.ranges                                      % Ranges to vary h
                args.writableVar (1, 1) double {mustBeNonnegative(args.writableVar)} = obj.writableVar                   % Indice of state variable to conduct analysys on
                args.maxValue (1, 1) double {mustBeNumeric(args.maxValue)} = obj.maxValue                                % Value to determine if system has diverged
                args.transient (1, 1) double {mustBeNonnegative(args.transient)} = obj.transient                         % Time to simulate transient
                args.values (1, :) double {mustBeNumeric(args.values)} = obj.values                                      % System parameters
                args.preScaller (1, 1) double {mustBeNonnegative(args.preScaller)} = obj.preScaller                      % Modifier to reduce computations (every <preScaller> point is computed)
            end
            arguments( Output )
                xData
                yData
            end

            % Call CUDA
            [xData, yData] = feval(sprintf('%s_bifurcation1DForHMEXWrapper', obj.SystemName), ...
                args.tMax, ...
                args.nPts, ...
                numel(args.intCon), ...
                args.intCon, ...
                args.ranges, ...
                args.writableVar, ...
                args.maxValue, ...
                args.transient, ...
                args.values, ...
                numel(args.values), ...
                args.preScaller);
        end

        function [xData, yData] = bifurcation1D(obj, args)
        %% bifurcation1D
        % Computes 1 dimensional bifurcation diagram
        % Inputs: 
        %   tMax (1, 1) double - Simultaion time
        %   nPts (1, 1) double - Number of point to analyze system in
        %   intCon (1, :) double - Initial conditions
        %   ranges (1, 2) double - Ranges to vary h
        %   writableVar (1, 1) double - Indice of state variable to conduct analysys on
        %   maxValue (1, 1) double - Value to determine if system has diverged
        %   transient (1, 1) double -Time to simulate transient
        %   values (1, :) double - System parameters
        %   preScaller (1, 1) double - Modifier to reduce computations (every <preScaller> point is computed)
        % Outputs:
        %   xData (1, N) double - Array of values for 1st mutVar
        %   yData (1, N) double - Array of amplitude peaks for writableVar 
            arguments( Input )
                obj (1, 1) CUDAHandler
                args.tMax (1, 1) double {mustBePositive(args.tMax)} = obj.tMax                                           % Simultaion time
                args.nPts (1, 1) double {mustBePositive(args.nPts)} = obj.nPts                                           % Number of point to analyze system in
                args.h (1, 1) double {mustBePositive(args.h)} = obj.h                                                    % Integration step
                args.intCon (1, :) double {mustBeNumeric(args.intCon)} = obj.intCon                                      % Initial conditions
                args.ranges (1, 2) double {mustBeNumeric(args.ranges)} = obj.ranges                                      % Ranges to vary h
                args.indicesOfMutVars (1, 1) double {mustBeNonnegative(args.indicesOfMutVars)} = obj.indicesOfMutVars    % Indices of mutVariabes
                args.writableVar (1, 1) double {mustBeNonnegative(args.writableVar)} = obj.writableVar                   % Indice of state variable to conduct analysys on
                args.maxValue (1, 1) double {mustBeNumeric(args.maxValue)} = obj.maxValue                                % Value to determine if system has diverged
                args.transient (1, 1) double {mustBeNonnegative(args.transient)} = obj.transient                         % Time to simulate transient
                args.values (1, :) double {mustBeNumeric(args.values)} = obj.values                                      % System parameters
                args.preScaller (1, 1) double {mustBeNonnegative(args.preScaller)} = obj.preScaller                      % Modifier to reduce computations (every <preScaller> point is computed)
            end
            arguments( Output )
                xData
                yData
            end

            % Call CUDA
            [xData, yData] = feval(sprintf('%s_bifurcation1DMEXWrapper', obj.SystemName), ...
                args.tMax, ...
                args.nPts, ...
                args.h, ...
                numel(args.intCon), ...
                args.intCon, ...
                args.ranges, ...
                args.indicesOfMutVars, ...
                args.writableVar, ...
                args.maxValue, ...
                args.transient, ...
                args.values, ...
                numel(args.values), ...
                args.preScaller);
        end

        function [xData, yData] = bifurcation1DIC(obj, args)
        %% bifurcation1DIC
        % Computes 1 dimensional bifurcation diagram for initial conditions
        % Inputs: 
        %   tMax (1, 1) double - Simultaion time
        %   nPts (1, 1) double - Number of point to analyze system in
        %   intCon (1, :) double - Initial conditions
        %   ranges (1, 2) double - Ranges to vary h
        %   writableVar (1, 1) double - Indice of state variable to conduct analysys on
        %   maxValue (1, 1) double - Value to determine if system has diverged
        %   transient (1, 1) double -Time to simulate transient
        %   values (1, :) double - System parameters
        %   preScaller (1, 1) double - Modifier to reduce computations (every <preScaller> point is computed)
        % Outputs:
        %   xData (1, N) double - Array of values for 1st mutVar
        %   yData (1, N) double - Array of amplitude peaks for writableVar 
            arguments( Input )
                obj (1, 1) CUDAHandler
                args.tMax (1, 1) double {mustBePositive(args.tMax)} = obj.tMax                                           % Simultaion time
                args.nPts (1, 1) double {mustBePositive(args.nPts)} = obj.nPts                                           % Number of point to analyze system in
                args.h (1, 1) double {mustBePositive(args.h)} = obj.h                                                    % Integration step
                args.intCon (1, :) double {mustBeNumeric(args.intCon)} = obj.intCon                                      % Initial conditions
                args.ranges (1, 2) double {mustBeNumeric(args.ranges)} = obj.ranges                                      % Ranges to vary h
                args.indicesOfMutVars (1, 1) double {mustBeNonnegative(args.indicesOfMutVars)} = obj.indicesOfMutVars    % Indices of mutVariabes
                args.writableVar (1, 1) double {mustBeNonnegative(args.writableVar)} = obj.writableVar                   % Indice of state variable to conduct analysys on
                args.maxValue (1, 1) double {mustBeNumeric(args.maxValue)} = obj.maxValue                                % Value to determine if system has diverged
                args.transient (1, 1) double {mustBeNonnegative(args.transient)} = obj.transient                         % Time to simulate transient
                args.values (1, :) double {mustBeNumeric(args.values)} = obj.values                                      % System parameters
                args.preScaller (1, 1) double {mustBeNonnegative(args.preScaller)} = obj.preScaller                      % Modifier to reduce computations (every <preScaller> point is computed)
            end
            arguments( Output )
                xData
                yData
            end

            % Call CUDA
            [xData, yData] = feval(sprintf('%s_bifurcation1DICMEXWrapper', obj.SystemName), ...
                args.tMax, ...
                args.nPts, ...
                args.h, ...
                numel(args.intCon), ...
                args.intCon, ...
                args.ranges, ...
                args.indicesOfMutVars, ...
                args.writableVar, ...
                args.maxValue, ...
                args.transient, ...
                args.values, ...
                numel(args.values), ...
                args.preScaller);
        end

        function [xData, yData, cData] = bifurcation2D(obj, args)
        %% bifurcation2D
        % Computes 2 dimensional bifurcation diagram
        % Input
        %   tMax (1, 1) double - Simultaion time
        %   nPts (1, 1) double - Number of point to analyze system in
        %   h (1, 1) double - Integration step
        %   intCon (1, :) double - Initial conditions
        %   ranges (2, 2) double - Ranges to vary h
        %   indicesOfMutVars (1, 1) double - Indices of mutVariabes
        %   writableVar (1, 1) double - Indice of state variable to conduct analysys on
        %   maxValue (1, 1) double - Value to determine if system has diverged
        %   transient (1, 1) double - Time to simulate transient
        %   values (1, :) double - System parameters
        %   preScaller (1, 1) double - Modifier to reduce computations (every <preScaller> point is computed)
        %   DBSCAN_eps (1, 1) double - Epsilon value for DBSCAN
        % Outputs 
        %   xData (1, N) double - Array of values for 1st mutVar
        %   yData (1, N) double - Array of values for 2st mutVar
        %   cData (N, N) double - Array of amplitude peaks for writableVar
            arguments( Input )
                obj (1, 1) CUDAHandler
                args.tMax (1, 1) double {mustBePositive(args.tMax)} = obj.tMax                                           % Simultaion time
                args.nPts (1, 1) double {mustBePositive(args.nPts)} = obj.nPts                                           % Number of point to analyze system in
                args.h (1, 1) double {mustBePositive(args.h)} = obj.h                                                    % Integration step
                args.intCon (1, :) double {mustBeNumeric(args.intCon)} = obj.intCon                                      % Initial conditions
                args.ranges (2, 2) double {mustBeNumeric(args.ranges)} = obj.ranges                                      % Ranges to vary h
                args.indicesOfMutVars (1, 1) double {mustBeNonnegative(args.indicesOfMutVars)} = obj.indicesOfMutVars    % Indices of mutVariabes
                args.writableVar (1, 1) double {mustBeNonnegative(args.writableVar)} = obj.writableVar                   % Indice of state variable to conduct analysys on
                args.maxValue (1, 1) double {mustBeNumeric(args.maxValue)} = obj.maxValue                                % Value to determine if system has diverged
                args.transient (1, 1) double {mustBeNonnegative(args.transient)} = obj.transient                         % Time to simulate transient
                args.values (1, :) double {mustBeNumeric(args.values)} = obj.values                                      % System parameters
                args.preScaller (1, 1) double {mustBeNonnegative(args.preScaller)} = obj.preScaller                      % Modifier to reduce computations (every <preScaller> point is computed)
                args.DBSCAN_eps (1, 1) double {mustBeNonnegative(args.DBSCAN_eps)} = obj.DBSCAN_eps                      % Epsilon value for DBSCAN
            end
            arguments( Output )
                xData
                yData
                cData
            end
            % Reshape ranges to CUDA style
            args.ranges = reshape(args.ranges', 1, []);

            % Call CUDA
            [xData, yData, cData] = feval(sprintf('%s_bifurcation2DMexWrapper', obj.SystemName), ...
                args.tMax, ...
                args.nPts, ...
                args.h, ...
                numel(args.intCon), ...
                args.intCon, ...
                args.ranges, ...
                args.indicesOfMutVars, ...
                args.writableVar, ...
                args.maxValue, ...
                args.transient, ...
                args.values, ...
                numel(args.values), ...
                args.preScaller, ...
                args.DBSCAN_eps);
        end

        function [xData, yData, cData] = bifurcation2DIC(obj, args)
        %% bifurcation2DIC
        % Computes 2 dimensional bifurcation diagram for initial conditions
        % Input
        %   tMax (1, 1) double - Simultaion time
        %   nPts (1, 1) double - Number of point to analyze system in
        %   h (1, 1) double - Integration step
        %   intCon (1, :) double - Initial conditions
        %   ranges (2, 2) double - Ranges to vary h
        %   indicesOfMutVars (1, 1) double - Indices of mutVariabes
        %   writableVar (1, 1) double - Indice of state variable to conduct analysys on
        %   maxValue (1, 1) double - Value to determine if system has diverged
        %   transient (1, 1) double - Time to simulate transient
        %   values (1, :) double - System parameters
        %   preScaller (1, 1) double - Modifier to reduce computations (every <preScaller> point is computed)
        %   DBSCAN_eps (1, 1) double - Epsilon value for DBSCAN
        % Outputs 
        %   xData (1, N) double - Array of values for 1st mutVar
        %   yData (1, N) double - Array of values for 2st mutVar
        %   cData (N, N) double - Array of amplitude peaks for writableVar
            arguments( Input )
                obj (1, 1) CUDAHandler
                args.tMax (1, 1) double {mustBePositive(args.tMax)} = obj.tMax                                           % Simultaion time
                args.nPts (1, 1) double {mustBePositive(args.nPts)} = obj.nPts                                           % Number of point to analyze system in
                args.h (1, 1) double {mustBePositive(args.h)} = obj.h                                                    % Integration step
                args.intCon (1, :) double {mustBeNumeric(args.intCon)} = obj.intCon                                      % Initial conditions
                args.ranges (2, 2) double {mustBeNumeric(args.ranges)} = obj.ranges                                      % Ranges to vary h
                args.indicesOfMutVars (1, 1) double {mustBeNonnegative(args.indicesOfMutVars)} = obj.indicesOfMutVars    % Indices of mutVariabes
                args.writableVar (1, 1) double {mustBeNonnegative(args.writableVar)} = obj.writableVar                   % Indice of state variable to conduct analysys on
                args.maxValue (1, 1) double {mustBeNumeric(args.maxValue)} = obj.maxValue                                % Value to determine if system has diverged
                args.transient (1, 1) double {mustBeNonnegative(args.transient)} = obj.transient                         % Time to simulate transient
                args.values (1, :) double {mustBeNumeric(args.values)} = obj.values                                      % System parameters
                args.preScaller (1, 1) double {mustBeNonnegative(args.preScaller)} = obj.preScaller                      % Modifier to reduce computations (every <preScaller> point is computed)
                args.DBSCAN_eps (1, 1) double {mustBeNonnegative(args.DBSCAN_eps)} = obj.DBSCAN_eps                      % Epsilon value for DBSCAN
            end
            arguments( Output )
                xData
                yData
                cData
            end
            % Reshape ranges to CUDA style
            args.ranges = reshape(args.ranges', 1, []);

            % Call CUDA
            [xData, yData, cData] = feval(sprintf('%s_bifurcation2DICMexWrapper', obj.SystemName), ...
                args.tMax, ...
                args.nPts, ...
                args.h, ...
                numel(args.intCon), ...
                args.intCon, ...
                args.ranges, ...
                args.indicesOfMutVars, ...
                args.writableVar, ...
                args.maxValue, ...
                args.transient, ...
                args.values, ...
                numel(args.values), ...
                args.preScaller, ...
                args.DBSCAN_eps);
        end

        function [xData, yData, cData] = LLE2DIC(obj, args)
        %% LLE2DIC
        % Computes 2 dimensional Lyapunov exponent diagram for initial
        % conditions
        % for varying parameters of system. It is much like 2D
        % bifurcation diagram, but bifurcation criteria is Lyapunov exponent
        % Inputs:
        %   tMax (1, 1) double -  Simultaion time
        %   NT (1, 1) double - Normaliztion time for LLE
        %   nPts (1, 1) double - Number of point to analyze system in
        %   h (1, 1) double - Integration step
        %   LLE_eps (1, 1) double - Epsilon value for LLE
        %   intCon (1, :) double - Initial conditions
        %   ranges (2, 2) double - Ranges to vary mutVariables
        %   indicesOfMutVars (1, 2) double - Indices of mutVariabes
        %   writableVar (1, 1) double - Indice of state variable to conduct analysis on
        %   maxValue (1, 1) double - Value to determine if system has diverged
        %   transient (1, 1) double - Time to simulate transient
        %   values (1, :) double - System parameters
        % Outputs:
        %   xData (1, N) double - Array of values for 1st mutVar
        %   yData (1, N) double - Array of values for 2nd mutVar
        %   cData (N, N) double - Matrix of Lyapunov exponent values
            arguments( Input )
                obj (1, 1) CUDAHandler
                args.tMax (1, 1) double {mustBePositive(args.tMax)} = obj.tMax                                           % Simultaion time
                args.NT (1, :) double {mustBePositive(args.NT)} = obj.NT                                                 % Normaliztion time for LLE
                args.nPts (1, 1) double {mustBePositive(args.nPts)} = obj.nPts                                           % Number of point to analyze system in
                args.h (1, 1) double {mustBePositive(args.h)} = obj.h                                                    % Integration step
                args.LLE_eps (1, 1) double {mustBePositive(args.LLE_eps)} = obj.LLE_eps                                  % Epsilon value for LLE
                args.intCon (1, :) double {mustBeNumeric(args.intCon)} = obj.intCon                                      % Initial conditions
                args.ranges (2, 2) double {mustBeNumeric(args.ranges)} = obj.ranges                                      % Ranges to vary mutVariables
                args.indicesOfMutVars (1, 2) double {mustBeNonnegative(args.indicesOfMutVars)} = obj.indicesOfMutVars    % Indices of mutVariabes
                args.writableVar (1, 1) double {mustBeNonnegative(args.writableVar)} = obj.writableVar                   % Indice of state variable to conduct analysys on
                args.maxValue (1, 1) double {mustBeNumeric(args.maxValue)} = obj.maxValue                                % Value to determine if system has diverged
                args.transient (1, 1) double {mustBeNonnegative(args.transient)} = obj.transient                         % Time to simulate transient
                args.values (1, :) double {mustBeNumeric(args.values)} = obj.values                                      % System parameters
            end
            arguments( Output )
                xData
                yData
                cData
            end
            % Reshape ranges to CUDA style
            args.ranges = reshape(args.ranges', 1, []);

            % Call CUDA
            [xData, yData, cData] = feval(sprintf('%s_LLE2DICMEXWrapper', obj.SystemName), ...
                args.tMax, ...
                args.NT, ...
                args.nPts, ...
                args.h, ...
                args.LLE_eps, ...
                args.intCon, ...
                numel(args.intCon), ...
                args.ranges, ...
                args.indicesOfMutVars, ...
                args.writableVar, ...
                args.maxValue, ...
                args.transient, ...
                args.values, ...
                numel(args.values));
        end

        function [xData, yData, cData] = LS2D(obj, args)
        %% LS2D 
        % Computes 2 dimensional LS metrics
        % Inputs:
        %   tMax (1, 1) double - Simultaion time
        %   NT (1, :) double - Normaliztion time
        %   nPts (1, 1) double - Number of point to analyze system in
        %   h (1, 1) double - Integration step
        %   LS_eps (1, 1) double - Epsilon value for LS
        %   intCon (1, :) double - Initial conditions
        %   ranges (2, 2) double - Ranges to vary mutVariables
        %   indicesOfMutVars (1, 2) double - Indices of mutVariabes
        %   writableVar (1, 1) double - Indice of state variable to conduct analysys on
        %   maxValue (1, 1) double - Value to determine if system has diverged
        %   transient (1, 1) double - Time to simulate transient
        %   values (1, :) double - System parameters
        % Outputs:
        %   xData (1, N) double - Array of values for 1st mutVar
        %   yData (1, N) double - Array of values for 2st mutVar
        %   cData (N, N) double - Matrix of LS Metrics
            arguments( Input )
                obj (1, 1) CUDAHandler
                args.tMax (1, 1) double {mustBePositive(args.tMax)} = obj.tMax                                           % Simultaion time
                args.NT (1, :) double {mustBePositive(args.NT)} = obj.NT                                                 % Normaliztion time
                args.nPts (1, 1) double {mustBePositive(args.nPts)} = obj.nPts                                           % Number of point to analyze system in
                args.h (1, 1) double {mustBePositive(args.h)} = obj.h                                                    % Integration step
                args.LS_eps (1, 1) double {mustBePositive(args.LS_eps)} = obj.LS_eps                                     % Epsilon value for LS
                args.intCon (1, :) double {mustBeNumeric(args.intCon)} = obj.intCon                                      % Initial conditions
                args.ranges (2, 2) double {mustBeNumeric(args.ranges)} = obj.ranges                                      % Ranges to vary mutVariables
                args.indicesOfMutVars (1, 2) double {mustBeNonnegative(args.indicesOfMutVars)} = obj.indicesOfMutVars    % Indices of mutVariabes
                args.writableVar (1, 1) double {mustBeNonnegative(args.writableVar)} = obj.writableVar                   % Indice of state variable to conduct analysys on
                args.maxValue (1, 1) double {mustBeNumeric(args.maxValue)} = obj.maxValue                                % Value to determine if system has diverged
                args.transient (1, 1) double {mustBeNonnegative(args.transient)} = obj.transient                         % Time to simulate transient
                args.values (1, :) double {mustBeNumeric(args.values)} = obj.values                                      % System parameters
            end
            arguments( Output )
                xData
                yData
                cData
            end
            % Reshape ranges to CUDA style
            args.ranges = reshape(args.ranges', 1, []);

            % Call CUDA
            [xData, yData, cData] = feval(sprintf('%s_LS2DMEXWrapper', obj.SystemName), ...
                args.tMax, ...
                args.NT, ...
                args.nPts, ...
                args.h, ...
                args.LS_eps, ...
                args.intCon, ...
                numel(args.intCon), ...
                args.ranges, ...
                args.indicesOfMutVars, ...
                args.writableVar, ...
                args.maxValue, ...
                args.transient, ...
                args.values, ...
                numel(args.values));
        end

        function [xData, yData] = LLE1DIC(obj, args)
        %% LLE1DIC
        % Computes 1 dimensional Lyapunov exponent diagram for initial conditions
        % Inputs
        %   tMax (1, 1) double - Simultaion time
        %   NT (1, :) double - Normaliztion time for LLE
        %   nPts (1, 1) double - Number of point to analyze system in
        %   h (1, 1) double - Integration step
        %   LLE_eps (1, 1) double - Epsilon value for LLE
        %   intCon (1, :) double - Initial conditions
        %   ranges (1, 2) double - Ranges to vary mutVariables
        %   indicesOfMutVars (1, 2) double - Indices of mutVariabes
        %   writableVar (1, 1) double - Indice of state variable to conduct analysys on
        %   maxValue (1, 1) double - Value to determine if system has diverged
        %   transient (1, 1) double - Time to simulate transient
        %   values (1, :) double - System parameters
        % Outputs
        %   xData (1, N) double - Array of values for 1st mutVar
        %   yData (1, N) double - Array of Lyapunov exponent values
            arguments( Input )
                obj (1, 1) CUDAHandler
                args.tMax (1, 1) double {mustBePositive(args.tMax)} = obj.tMax                                           % Simultaion time
                args.NT (1, :) double {mustBePositive(args.NT)} = obj.NT                                                 % Normaliztion time for LLE
                args.nPts (1, 1) double {mustBePositive(args.nPts)} = obj.nPts                                           % Number of point to analyze system in
                args.h (1, 1) double {mustBePositive(args.h)} = obj.h                                                    % Integration step
                args.LLE_eps (1, 1) double {mustBePositive(args.LLE_eps)} = obj.LLE_eps                                  % Epsilon value for LLE
                args.intCon (1, :) double {mustBeNumeric(args.intCon)} = obj.intCon                                      % Initial conditions
                args.ranges (1, 2) double {mustBeNumeric(args.ranges)} = obj.ranges                                      % Ranges to vary mutVariables
                args.indicesOfMutVars (1, 2) double {mustBeNonnegative(args.indicesOfMutVars)} = obj.indicesOfMutVars    % Indices of mutVariabes
                args.writableVar (1, 1) double {mustBeNonnegative(args.writableVar)} = obj.writableVar                   % Indice of state variable to conduct analysys on
                args.maxValue (1, 1) double {mustBeNumeric(args.maxValue)} = obj.maxValue                                % Value to determine if system has diverged
                args.transient (1, 1) double {mustBeNonnegative(args.transient)} = obj.transient                         % Time to simulate transient
                args.values (1, :) double {mustBeNumeric(args.values)} = obj.values                                      % System parameters
            end
            arguments( Output )
                xData
                yData
            end

            % Call CUDA
            [xData, yData] = feval(sprintf('%s_LLE1DICMEXWrapper', obj.SystemName), ...
                args.tMax, ...
                args.NT, ...
                args.nPts, ...
                args.h, ...
                args.LLE_eps, ...
                args.intCon, ...
                numel(args.intCon), ...
                args.ranges, ...
                args.indicesOfMutVars, ...
                args.writableVar, ...
                args.maxValue, ...
                args.transient, ...
                args.values, ...
                numel(args.values));
        end

        function [xData, yData] = LLE1D(obj, args)
        %% LLE1DIC
        % Computes 1 dimensional Lyapunov exponent diagram
        % Inputs
        %   tMax (1, 1) double - Simultaion time
        %   NT (1, :) double - Normaliztion time for LLE
        %   nPts (1, 1) double - Number of point to analyze system in
        %   h (1, 1) double - Integration step
        %   LLE_eps (1, 1) double - Epsilon value for LLE
        %   intCon (1, :) double - Initial conditions
        %   ranges (1, 2) double - Ranges to vary mutVariables
        %   indicesOfMutVars (1, 2) double - Indices of mutVariabes
        %   writableVar (1, 1) double - Indice of state variable to conduct analysys on
        %   maxValue (1, 1) double - Value to determine if system has diverged
        %   transient (1, 1) double - Time to simulate transient
        %   values (1, :) double - System parameters
        % Outputs
        %   xData (1, N) double - Array of values for 1st mutVar
        %   yData (1, N) double - Array of Lyapunov exponent values
            arguments( Input )
                obj (1, 1) CUDAHandler
                args.tMax (1, 1) double {mustBePositive(args.tMax)} = obj.tMax                                           % Simultaion time
                args.NT (1, :) double {mustBePositive(args.NT)} = obj.NT                                                 % Normaliztion time for LLE
                args.nPts (1, 1) double {mustBePositive(args.nPts)} = obj.nPts                                           % Number of point to analyze system in
                args.h (1, 1) double {mustBePositive(args.h)} = obj.h                                                    % Integration step
                args.LLE_eps (1, 1) double {mustBePositive(args.LLE_eps)} = obj.LLE_eps                                  % Epsilon value for LLE
                args.intCon (1, :) double {mustBeNumeric(args.intCon)} = obj.intCon                                      % Initial conditions
                args.ranges (1, 2) double {mustBeNumeric(args.ranges)} = obj.ranges                                      % Ranges to vary mutVariables
                args.indicesOfMutVars (1, 2) double {mustBeNonnegative(args.indicesOfMutVars)} = obj.indicesOfMutVars    % Indices of mutVariabes
                args.writableVar (1, 1) double {mustBeNonnegative(args.writableVar)} = obj.writableVar                   % Indice of state variable to conduct analysys on
                args.maxValue (1, 1) double {mustBeNumeric(args.maxValue)} = obj.maxValue                                % Value to determine if system has diverged
                args.transient (1, 1) double {mustBeNonnegative(args.transient)} = obj.transient                         % Time to simulate transient
                args.values (1, :) double {mustBeNumeric(args.values)} = obj.values                                      % System parameters
            end
            arguments( Output )
                xData
                yData
            end

            % Call CUDA
            [xData, yData] = feval(sprintf('%s_LLE1DMEXWrapper', obj.SystemName), ...
                args.tMax, ...
                args.NT, ...
                args.nPts, ...
                args.h, ...
                args.LLE_eps, ...
                args.intCon, ...
                numel(args.intCon), ...
                args.ranges, ...
                args.indicesOfMutVars, ...
                args.writableVar, ...
                args.maxValue, ...
                args.transient, ...
                args.values, ...
                numel(args.values));
        end

        function [xData, yData] = LS1D(obj, args)
        %% LS1D
        % Computes 1 dimentional LS metrics
        % Inputs
        %   tMax (1, 1) double - Simultaion time
        %   NT (1, :) double - Normaliztion time
        %   nPts (1, 1) double - Number of point to analyze system in
        %   h (1, 1) double - Integration step
        %   LS_eps (1, 1) double - Epsilon value for LS
        %   intCon (1, :) double - Initial conditions
        %   ranges (1, 2) double - Ranges to vary mutVariables
        %   indicesOfMutVars (1, 2) double - Indices of mutVariabes
        %   writableVar (1, 1) double - Indice of state variable to conduct analysys on
        %   maxValue (1, 1) double - Value to determine if system has diverged
        %   transient (1, 1) double - Time to simulate transient
        %   values (1, :) double - System parameters
        % Outputs:
        %   xData (1, N) double - Array of values for 1st mutVar
        %   yData (1, N) double - Array for LS Metrics
            arguments( Input )
                obj (1, 1) CUDAHandler
                args.tMax (1, 1) double {mustBePositive(args.tMax)} = obj.tMax                                           % Simultaion time
                args.NT (1, :) double {mustBePositive(args.NT)} = obj.NT                                                 % Normaliztion time
                args.nPts (1, 1) double {mustBePositive(args.nPts)} = obj.nPts                                           % Number of point to analyze system in
                args.h (1, 1) double {mustBePositive(args.h)} = obj.h                                                    % Integration step
                args.LS_eps (1, 1) double {mustBePositive(args.LS_eps)} = obj.LS_eps                                     % Epsilon value for LS
                args.intCon (1, :) double {mustBeNumeric(args.intCon)} = obj.intCon                                      % Initial conditions
                args.ranges (1, 2) double {mustBeNumeric(args.ranges)} = obj.ranges                                      % Ranges to vary mutVariables
                args.indicesOfMutVars (1, 2) double {mustBeNonnegative(args.indicesOfMutVars)} = obj.indicesOfMutVars    % Indices of mutVariabes
                args.writableVar (1, 1) double {mustBeNonnegative(args.writableVar)} = obj.writableVar                   % Indice of state variable to conduct analysys on
                args.maxValue (1, 1) double {mustBeNumeric(args.maxValue)} = obj.maxValue                                % Value to determine if system has diverged
                args.transient (1, 1) double {mustBeNonnegative(args.transient)} = obj.transient                         % Time to simulate transient
                args.values (1, :) double {mustBeNumeric(args.values)} = obj.values                                      % System parameters
            end
            arguments( Output )
                xData
                yData
            end

            % Call CUDA
            [xData, yData] = feval(sprintf('%s_LS1DMEXWrapper', obj.SystemName), ...
                args.tMax, ...
                args.NT, ...
                args.nPts, ...
                args.h, ...
                args.LS_eps, ...
                args.intCon, ...
                numel(args.intCon), ...
                args.ranges, ...
                args.indicesOfMutVars, ...
                args.writableVar, ...
                args.maxValue, ...
                args.transient, ...
                args.values, ...
                numel(args.values));
        end
    end
    methods(Static)
            function gOpts()
                %% Prints all global options to console
                % You can easily copy paste them into any file to use
                % Just replace <class obj name> with actual name of
                % CUDAHandler object and set desired values
                type gDat.txt
            end
    end
end