addpath('/media/d/workspace/common/matlab');
addpath('topictoolbox/');

if (1)
    kmer_hist = load('../result/kmer_hist.txt');
    [m, n] = size(kmer_hist);
end

stmp = mean(kmer_hist,2);
idx = (stmp < 30) & (stmp > 2);
kmer_hist = kmer_hist(idx,:);

data = kmer_hist(:,1:10:end);
data1 = kmer_hist(:,2:10:end);

% keyboard;

X = data';
[di, wj, c] = find(X); 
WS = zeros(1,sum(c));
DS = zeros(1,sum(c));
idx = 0;
for i = 1:length(c)
    WS(idx+(1:c(i))) = wj(i);
    DS(idx+(1:c(i))) = di(i);
    idx = idx + c(i);
end

X1 = data1';
[di, wj, c] = find(X1); 
WS1 = zeros(1,sum(c));
DS1 = zeros(1,sum(c));
idx = 0;
for i = 1:length(c)
    WS1(idx+(1:c(i))) = wj(i);
    DS1(idx+(1:c(i))) = di(i);
    idx = idx + c(i);
end

[D, W] = size(X);
T = 20; 
N = 1000; 
BETA = 200 / W; 
ALPHA = 50 / T;

SEED = 3;
OUTPUT = 1;

% tic
% [ WP,DP,Z ] = GibbsSamplerLDA( WS , DS , T , N , ALPHA , BETA , SEED , OUTPUT );
% toc
tic
[ WP0,DP0,Z0,WP1,DP1,Z1 ] = GibbsSamplerLDA_bilingual( WS, DS, T, N, ALPHA, BETA, ...
    SEED ,OUTPUT, WS1, DS1 );
toc

SEED = 3;
OUTPUT = 1;

WS_cell = {WS, WS1};
DS_cell = {DS, DS1};
tic
[WP, DP, Z] = GibbsSamplerLDA_polylingual(WS_cell, DS_cell, T, N, ALPHA, BETA, ...
    SEED ,OUTPUT);
toc

WP = reshape(WP, T, []);
DP = reshape(DP, T, []);

WP0_ = WP(:,1:W)';
WP1_ = WP(:,W+(1:W))';
DP0_ = DP(:,1:D)';
DP1_ = DP(:,D+(1:D))';

[r, p] = corrcoef(full([WP0, WP1]));
figure; imagesc(log10(p) < -5);

[r1, p1] = corrcoef(full([WP0_, WP1_]));
figure; imagesc(log10(p1) < -5);

tmp = r(1:20,21:40);
tmp1 = r1(1:20,21:40);
[ss, si] = sort(diag(tmp));
[ss1, si1] = sort(diag(tmp1));

[r2, p2] = corrcoef(full([WP0(:,si), WP0_(:,si1)]));
figure; imagesc(r2);

return