function [f]=fitness_nq(solution)
% returns the amount of collisions for a given solution (permutation).
% the maximum amount of collisions that can occur is n(n-1)/2.
% for a 4x4 chessboard, the maximum amount of collisions is 6, corresponding 
% to the situation in which all queens are in the same diagonal.
    f=0;
    n=length(solution);
    for i=1:n
        for j=1:n
            if abs(i-j)==abs(solution(i)-solution(j)) && i~=j,
                f=f+1;               
            end
        end
    end
    f=f/2;  
end