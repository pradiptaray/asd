#include "mex.h"
#include "cokus.cpp"

// Syntax
//   [ WP , DP , Z ] = GibbsSamplerLDA( WS , DS , T , N , ALPHA , BETA , SEED , OUTPUT )

// Syntax
//   [ WP , DP , Z ] = GibbsSamplerLDA( WS , DS , T , N , ALPHA , BETA , SEED , OUTPUT , ZIN )


void GibbsSamplerLDA(double ALPHA, double BETA, int W, int T, int D, int NN, int OUTPUT, int n, int *z, int *d, int *w, int *wp, int *dp, int *ztot, int *order, double *probs, int WS_no, int *ntokens_count, int* language)
{
  int wi,di,i,ii,j,jj,ll,topic, rp, temp, iter, wioffset, dioffset;
  double totprob, WBETA, r, max, totaldeno;
  
  /* random initialization */
  if (OUTPUT==2) mexPrintf( "Starting Random initialization\n" );
  
  n = 0;
  for (i = 0; i < WS_no; i++)
  {
    for (j=0; j < ntokens_count[i]; j++)
    {
      wi = w[j+n]; 
      di = d[j+n];
      
      // pick a random topic 0..T-1
      topic = (int) ( (double) randomMT() * (double) T / (double) (4294967296.0 + 1.0) );
      z[j+n] = topic; // assign this word token to this topic
      wp[i*W*T + wi*T + topic]++; // increment wp count matrix
      dp[i*D*T + di*T + topic]++; // increment dp count matrix
      ztot[i*T + topic]++; // increment ztot matrix      
    }
    
    n += ntokens_count[i];
  }
      
  if (OUTPUT==2) mexPrintf( "Determining random order update sequence\n" );
  
  for (i = 0; i < n; i++) order[i] = i; // fill with increasing series
  
  n = 0;
  for (i = 0; i < WS_no; i++)
  {
    for (j = 0; j < ntokens_count[i] - 1; j++)
    {
      // pick a random integer between i and nw
      rp = j + (int) ((double) (ntokens_count[i]-j) * (double) randomMT() / (double) (4294967296.0 + 1.0));
      
      // switch contents on position i and position rp
      temp = order[n + rp];
      order[n + rp] = order[n + j];
      order[n + j] = temp;      
    }
    
    n += ntokens_count[i];    
  }
  
  /*
  for (i=0; i < (n-1); i++) {
      // pick a random integer between i and nw
      rp = i + (int) ((double) (n-i) * (double) randomMT() / (double) (4294967296.0 + 1.0));
      
      // switch contents on position i and position rp
      temp = order[rp];
      order[rp]=order[i];
      order[i]=temp;
  }
  */
      
  //for (i=0; i<n; i++) mexPrintf( "i=%3d order[i]=%3d\n" , i , order[ i ] );
  
  for (iter=0; iter < NN; iter++) 
  {
      if (OUTPUT >= 1) {
          if ((iter % 1)==0) mexPrintf( "\tIteration %d of %d\n" , iter , NN );
          if ((iter % 1)==0) mexEvalString("drawnow;");
      }
      
      for (ii = 0; ii < n; ii++)
      {
        i = order[ii]; // current word token to assess
                                
        wi  = w[i]; // current word index
        di  = d[i]; // current document index  
        topic = z[i]; // current topic assignment to word token
        ll = language[i];
        
        ztot[ll*T + topic]--;  // substract this from counts
        
        wioffset = ll*W*T + wi*T;
        dioffset = ll*D*T + di*T;
        
        wp[wioffset +  topic]--;
        dp[dioffset + topic]--;
        
        // mexPrintf( "(1) Working on ii=%d i=%d wi=%d di=%d topic=%d wp=%d dp=%d\n" , ii , i , wi , di , topic , wp[wi+topic*W] , dp[wi+topic*D] );
        // mexEvalString("drawnow;");
        
        totprob = (double) 0;
        for (j = 0; j < T; j++) 
        {
          totaldeno = (double) 0;
          for (jj = 0; jj < WS_no; jj++)
            totaldeno += (double) dp[language[jj]*D*T + di*T + j];
            
          probs[j] = ((double) wp[ wioffset+j ] + (double) BETA) / ((double) ztot[ll*T + j] + (double) (W*BETA)) * (totaldeno + (double) ALPHA);
            totprob += probs[j];
        }
        
        // sample a topic from the distribution
        r = (double) totprob * (double) randomMT() / (double) 4294967296.0;
        max = probs[0];
        topic = 0;
        while (r>max) {
            topic++;
            max += probs[topic];
        }
         
        z[i] = topic; // assign current word token i to topic j
        wp[wioffset + topic]++; // and update counts
        dp[dioffset + topic]++;
        ztot[ll*T + topic]++;        
      }
  }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                 const mxArray *prhs[])
{
  int WS_no, DS_no;
  mxArray *WS_cell, *DS_cell;
  
  double *srwp, *srdp, *probs, *Z, *WS, *DS;
  double ALPHA,BETA;
  size_t *irwp, *jcwp, *irdp, *jcdp;
  int *z,*d,*w, *order, *wp, *dp, *ztot, *language;
  int W,T,D,NN,SEED,OUTPUT, nzmax, nzmaxwp, nzmaxdp, ntokens, *ntokens_count;
  int i,j,c,n,nt,wi,di, startcond;
  
  /* Check for proper number of arguments. */
  if (nrhs < 8) {
    mexErrMsgTxt("At least 8 input arguments required");
  } else if (nlhs < 3) {
    mexErrMsgTxt("3 output arguments required");
  }  
  
  /* process the input arguments */
  if (mxIsCell(prhs[0]) != 1) mexErrMsgTxt("WS input vector must be a cell array of matrices");
  if (mxIsCell(prhs[1]) != 1) mexErrMsgTxt("DS input vector must be a cell array of matrices");
  
  WS_no = (int) mxGetNumberOfElements(prhs[0]);
  DS_no = (int) mxGetNumberOfElements(prhs[1]);
  if (WS_no != DS_no) mexErrMsgTxt("WS number not equal to DS number");     
  
  T    = (int) mxGetScalar(prhs[2]);
  if (T<=0) mexErrMsgTxt("Number of topics must be greater than zero");
  
  NN    = (int) mxGetScalar(prhs[3]);
  if (NN<0) mexErrMsgTxt("Number of iterations must be positive");
  
  ALPHA = (double) mxGetScalar(prhs[4]);
  if (ALPHA<=0) mexErrMsgTxt("ALPHA must be greater than zero");
  
  BETA = (double) mxGetScalar(prhs[5]);
  if (BETA<=0) mexErrMsgTxt("BETA must be greater than zero");
  
  SEED = (int) mxGetScalar(prhs[6]);
  
  OUTPUT = (int) mxGetScalar(prhs[7]);
    
  /* seeding */
  seedMT( 1 + SEED * 2 ); // seeding only works on uneven numbers  

  // allocate memory;  
  ztot  = (int *) mxCalloc(T*WS_no, sizeof(int));  
  probs  = (double *) mxCalloc(T, sizeof(double));  
  ntokens_count = (int *) mxCalloc(WS_no, sizeof(int));
  ntokens = 0;
    
  // process cell array;    
  for (i = 0; i < WS_no; i++)
  {
    WS_cell = mxGetCell(prhs[0], i);
    DS_cell = mxGetCell(prhs[1], i);
    
    WS = mxGetPr(WS_cell); 
    DS = mxGetPr(DS_cell);
    
    n = mxGetM(WS_cell) * mxGetN(WS_cell);
    
    if (n == 0) mexErrMsgTxt("WS vector is empty"); 
    if (n != (mxGetM(DS_cell) * mxGetN(DS_cell))) mexErrMsgTxt("WS and DS vectors should have same number of entries");
    
    ntokens_count[i] = n;
    ntokens += n;                       
  }     
  
  /* allocate memory */
  z  = (int *) mxCalloc(ntokens, sizeof(int));
    
  d  = (int *) mxCalloc(ntokens, sizeof(int));
  w  = (int *) mxCalloc(ntokens, sizeof(int));
  order  = (int *) mxCalloc(ntokens, sizeof(int));  
  language  = (int *) mxCalloc(ntokens, sizeof(int));
  
  n = 0;
  W = 0;
  D = 0;
  for (i = 0; i < WS_no; i++)
  {
    WS_cell = mxGetCell(prhs[0], i);
    DS_cell = mxGetCell(prhs[1], i);
    
    WS = mxGetPr(WS_cell); 
    DS = mxGetPr(DS_cell);

    // copy over the word and document indices into internal format    
    for (j = 0; j < ntokens_count[i]; j++)
    {
      w[j + n] = (int) WS[j] - 1;
      d[j + n] = (int) DS[j] - 1;    
      language[j + n] = i;  
      
      if (w[j + n] > W) W = w[j + n];
      if (d[j + n] > D) D = d[j + n];
    }   
    
    n += ntokens_count[i]; 
  }  
  W = W + 1;
  D = D + 1;  
    
  wp  = (int *) mxCalloc(T*W*WS_no, sizeof(int));
  dp  = (int *) mxCalloc(T*D*WS_no, sizeof(int));
       
  //mexPrintf( "N=%d  T=%d W=%d D=%d\n" , ntokens , T , W , D );
  
  if (OUTPUT==2) {
      mexPrintf( "Running LDA Gibbs Sampler Version 1.0\n" );

      mexPrintf( "Arguments:\n" );
      mexPrintf( "\tNumber of words      W = %d\n"    , W );
      mexPrintf( "\tNumber of docs       D = %d\n"    , D );
      mexPrintf( "\tNumber of topics     T = %d\n"    , T );
      mexPrintf( "\tNumber of iterations N = %d\n"    , NN );
      mexPrintf( "\tHyperparameter   ALPHA = %4.4f\n" , ALPHA );
      mexPrintf( "\tHyperparameter    BETA = %4.4f\n" , BETA );
      mexPrintf( "\tSeed number            = %d\n"    , SEED );
      mexPrintf( "\tNumber of tokens       = %d\n"    , ntokens );
      mexPrintf( "Internal Memory Allocation\n" );
      mexPrintf( "\tw,d,z,order indices combined = %d bytes\n" , 4 * sizeof( int) * ntokens );
      mexPrintf( "\twp (full) matrix = %d bytes\n" , sizeof( int ) * W * T  );
      mexPrintf( "\tdp (full) matrix = %d bytes\n" , sizeof( int ) * D * T  );
      //mexPrintf( "Checking: sizeof(int)=%d sizeof(long)=%d sizeof(double)=%d\n" , sizeof(int) , sizeof(long) , sizeof(double));
  }
  
  /* run the model */
  GibbsSamplerLDA(ALPHA, BETA, W, T, D, NN, OUTPUT, ntokens, z, d, w, wp, dp, ztot, order, probs, WS_no, ntokens_count, language);
  
  /* convert the full wp matrix into a sparse matrix */
  nzmaxwp = 0;
  for (i=0; i<W; i++) {
     for (j=0; j<T; j++)
         nzmaxwp += (int) ( *( wp + j + i*T )) > 0;
  }  
   
  if (OUTPUT==2) {
      mexPrintf( "Constructing sparse output matrix wp\n" );
      mexPrintf( "Number of nonzero entries for WP = %d\n" , nzmaxwp );
  }

  /*  
  // MAKE THE WP SPARSE MATRIX
  plhs[0] = mxCreateSparse( W,T,nzmaxwp,mxREAL);
  srwp  = mxGetPr(plhs[0]);
  irwp = mxGetIr(plhs[0]);
  jcwp = mxGetJc(plhs[0]);  
  n = 0;
  for (j=0; j<T; j++) {
      *( jcwp + j ) = n;
      for (i=0; i<W; i++) {
         c = (int) *( wp + i*T + j );
         if (c >0) {
             *( srwp + n ) = c;
             *( irwp + n ) = i;
             n++;
         }
      }    
  }  
  *( jcwp + T ) = n;    
    
  // MAKE THE DP SPARSE MATRIX
  nzmaxdp = 0;
  for (i=0; i<D; i++) {
      for (j=0; j<T; j++)
          nzmaxdp += (int) ( *( dp + j + i*T )) > 0;
  }  
   
  if (OUTPUT==2) {
      mexPrintf( "Constructing sparse output matrix dp\n" );
      mexPrintf( "Number of nonzero entries for DP = %d\n" , nzmaxdp );
  }
  
  plhs[1] = mxCreateSparse( D,T,nzmaxdp,mxREAL);
  srdp  = mxGetPr(plhs[1]);
  irdp = mxGetIr(plhs[1]);
  jcdp = mxGetJc(plhs[1]);
  n = 0;
  for (j=0; j<T; j++) {
      *( jcdp + j ) = n;
      for (i=0; i<D; i++) {
          c = (int) *( dp + i*T + j );
          if (c >0) {
              *( srdp + n ) = c;
              *( irdp + n ) = i;
              n++;
          }
      }
  }
  *( jcdp + T ) = n;
  */
  
  plhs[0] = mxCreateDoubleMatrix(1, WS_no*W*T, mxREAL);
  srwp = mxGetPr(plhs[0]);
  for (i = 0; i < WS_no*W*T; i++) srwp[i] = (double) wp[i];
  
  plhs[1] = mxCreateDoubleMatrix(1, WS_no*D*T, mxREAL);
  srdp = mxGetPr(plhs[1]);
  for (i = 0; i < WS_no*D*T; i++) srdp[i] = (double) dp[i];
   
  plhs[2] = mxCreateDoubleMatrix(1, ntokens , mxREAL);
  Z = mxGetPr(plhs[2]);
  for (i = 0; i < ntokens; i++) Z[i] = (double) z[i] + 1;  
}
