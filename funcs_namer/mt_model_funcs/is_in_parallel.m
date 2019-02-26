function answer = is_in_parallel ()
    try
        answer = ~isempty(getCurrentTask());
    catch err
        if ~strcmp(err.identifier, 'MATLAB:UndefinedFunction')
            rethrow(err);
        end
        answer = false;
    end
end