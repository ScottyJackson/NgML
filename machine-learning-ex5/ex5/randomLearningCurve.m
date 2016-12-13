function [error_train, error_val] = ...
    randomLearningCurve(X, y, Xval, yval, lambda, n_iter)

m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i = 1:m
    fprintf('ITERATION %d\n', i);
    train_sum = 0;
    val_sum = 0;
    for j = 1:n_iter
        % take random sample of train/val sets
        train_samp = randsample(size(X,1), i);
        val_samp = randsample(size(Xval,1), i);
        X_samp = X(train_samp,:);
        y_samp = y(train_samp);
        Xval_samp = Xval(val_samp,:);
        yval_samp = yval(val_samp);
        
        theta = trainLinearReg(X_samp, y_samp, lambda);
        train_sum = train_sum + linearRegCostFunction(X_samp, y_samp, theta, 0);
        val_sum = val_sum + linearRegCostFunction(Xval_samp, yval_samp, theta, 0);
    end
    error_train(i) = train_sum / n_iter;
    error_val(i) = val_sum / n_iter;
end

end

