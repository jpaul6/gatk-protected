/*
* Copyright 2012-2015 Broad Institute, Inc.
* 
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without
* restriction, including without limitation the rights to use,
* copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following
* conditions:
* 
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
* HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
* THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

package org.broadinstitute.gatk.utils;

import cern.jet.math.Arithmetic;
import cern.jet.random.Normal;
import com.google.java.contract.Ensures;
import com.google.java.contract.Requires;
import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.NormalDistribution;
import org.apache.commons.math.distribution.NormalDistributionImpl;
import org.broadinstitute.gatk.utils.collections.Pair;
import org.broadinstitute.gatk.utils.exceptions.GATKException;

import java.io.Serializable;
import java.util.*;

/**
 * Created by IntelliJ IDEA.
 * User: chartl
 */
public class MannWhitneyU {

    private static Normal STANDARD_NORMAL = new Normal(0.0,1.0,null);
    private static NormalDistribution APACHE_NORMAL = new NormalDistributionImpl(0.0,1.0,1e-2);
    private static double LNSQRT2PI = Math.log(Math.sqrt(2.0*Math.PI));

    private TreeMap<Number, Pair<Integer, Integer>> observations;
    private int sizeSet1;
    private int sizeSet2;
    private ExactMode exactMode;

    public MannWhitneyU(ExactMode mode, boolean dither) {
        observations = new TreeMap<>();
        sizeSet1 = 0;
        sizeSet2 = 0;
        exactMode = mode;
    }

    public MannWhitneyU() {
        this(ExactMode.POINT,true);
    }

    public MannWhitneyU(boolean dither) {
        this(ExactMode.POINT,dither);
    }

    public MannWhitneyU(ExactMode mode) {
        this(mode,true);
    }

    /**
     * Add an observation into the observation tree
     * @param n: the observation (a number)
     * @param set: whether the observation comes from set 1 or set 2
     */
    public void add(Number n, USet set) {
        Pair<Integer, Integer> currValue = observations.get(n);
        if (currValue == null) {
            currValue = new Pair<>(0, 0);
            observations.put(n, currValue);
        }

        if (set == USet.SET1) {
            currValue.first += 1;
        } else {
            currValue.second += 1;
        }

        if ( set == USet.SET1 ) {
            ++sizeSet1;
        } else {
            ++sizeSet2;
        }
    }

    public Pair<Double, Double> getR1R2() {
        double u1 = calculateOneSidedU(observations,MannWhitneyU.USet.SET1);
        long n1 = sizeSet1*(sizeSet1+1)/2;
        double r1 = u1 + n1;
        long n2 = sizeSet2*(sizeSet2+1)/2;
        double u2 = n1*n2-u1;
        double r2 = u2 + n2;

        return new Pair<Double,Double>(r1,r2);
    }

    /**
     * Runs the one-sided test under the hypothesis that the data in set "lessThanOther" stochastically
     * dominates the other set
     * @param lessThanOther - either Set1 or Set2
     * @return - u-based z-approximation, and p-value associated with the test (p-value is exact for small n,m)
     */
    @Requires({"lessThanOther != null"})
    @Ensures({"validateObservations(observations) || Double.isNaN(result.getFirst())","result != null", "! Double.isInfinite(result.getFirst())", "! Double.isInfinite(result.getSecond())"})
    public Pair<Double,Double> runOneSidedTest(USet lessThanOther) {
        double u = calculateOneSidedU(observations, lessThanOther);
        Map<Integer, Integer> tieStructure = getTieStructure(observations);
        int n = lessThanOther == USet.SET1 ? sizeSet1 : sizeSet2;
        int m = lessThanOther == USet.SET1 ? sizeSet2 : sizeSet1;
        if ( n == 0 || m == 0 ) {
            // test is uninformative as one or both sets have no observations
            return new Pair<Double,Double>(Double.NaN,Double.NaN);
        }

        // the null hypothesis is that {N} is stochastically less than {M}, so U has counted
        // occurrences of {M}s before {N}s. We would expect that this should be less than (n*m+1)/2 under
        // the null hypothesis, so we want to integrate from K=0 to K=U for cumulative cases. Always.
        return calculateP(n, m, u, false, tieStructure, exactMode);
    }

    /**
     * Runs the standard two-sided test,
     * returns the u-based z-approximate and p values.
     * @return a pair holding the u and p-value.
     */
    @Ensures({"result != null", "! Double.isInfinite(result.getFirst())", "! Double.isInfinite(result.getSecond())"})
    //@Requires({"validateObservations(observations)"})
    public Pair<Double,Double> runTwoSidedTest() {
        Pair<Double,USet> uPair = calculateTwoSidedU(observations);
        double u = uPair.first;
        Map<Integer, Integer> tieStructure = getTieStructure(observations);
        int n = uPair.second == USet.SET1 ? sizeSet1 : sizeSet2;
        int m = uPair.second == USet.SET1 ? sizeSet2 : sizeSet1;
        if ( n == 0 || m == 0 ) {
            // test is uninformative as one or both sets have no observations
            return new Pair<Double,Double>(Double.NaN,Double.NaN);
        }
        return calculateP(n, m, u, true, tieStructure, exactMode);
    }

    /**
     * Given a u statistic, calculate the p-value associated with it, dispatching to approximations where appropriate
     * @param n - The number of entries in the stochastically smaller (dominant) set
     * @param m - The number of entries in the stochastically larger (dominated) set
     * @param u - the Mann-Whitney U value
     * @param twoSided - is the test twosided
     * @return the (possibly approximate) p-value associated with the MWU test, and the (possibly approximate) z-value associated with it
     * todo -- there must be an approximation for small m and large n
     */
    @Requires({"m > 0","n > 0"})
    @Ensures({"result != null", "! Double.isInfinite(result.getFirst())", "! Double.isInfinite(result.getSecond())"})
    protected static Pair<Double,Double> calculateP(int n, int m, double u, boolean twoSided, Map<Integer, Integer> tieStructure, ExactMode exactMode) {
        Pair<Double,Double> zandP;
        if ( n > 8 && m > 8 ) {
            // large m and n - normal approx
            zandP = calculatePNormalApproximation(n,m,u,tieStructure, twoSided);
        } else if ( n > 5 && m > 7 ) {
            // large m, small n - sum uniform approx
            // todo -- find the appropriate regimes where this approximation is actually better enough to merit slowness
            // pval = calculatePUniformApproximation(n,m,u);
            zandP = calculatePNormalApproximation(n, m, u, tieStructure, twoSided);
        } else if ( n > 8 || m > 8 ) {
            zandP = calculatePFromTable(n, m, u, tieStructure, twoSided);
        } else {
            // small m and n - full approx
            zandP = calculatePRecursively(n,m,u,twoSided,exactMode);
        }

        return zandP;
    }

    public static Pair<Double,Double> calculatePFromTable(int n, int m, double u, Map<Integer, Integer> tieStructure, boolean twoSided) {
        // todo -- actually use a table for:
        // todo      - n large, m small
        return calculatePNormalApproximation(n,m,u, tieStructure, twoSided);
    }

    /**
     * Uses a normal approximation to the U statistic in order to return a cdf p-value. See Mann, Whitney [1947]
     * @param n - The number of entries in the stochastically smaller (dominant) set
     * @param m - The number of entries in the stochastically larger (dominated) set
     * @param u - the Mann-Whitney U value
     * @param twoSided - whether the test should be two sided
     * @return p-value associated with the normal approximation
     */
    @Requires({"m > 0","n > 0"})
    @Ensures({"result != null", "! Double.isInfinite(result.getFirst())", "! Double.isInfinite(result.getSecond())"})
    public static Pair<Double,Double> calculatePNormalApproximation(int n,int m,double u, Map<Integer, Integer> tieStructure, boolean twoSided) {
        double z = getZApprox(n,m,u,tieStructure);
        if ( twoSided ) {
            return new Pair<Double,Double>(z,2.0*(z < 0 ? STANDARD_NORMAL.cdf(z) : 1.0-STANDARD_NORMAL.cdf(z)));
        } else {
            return new Pair<Double,Double>(z,STANDARD_NORMAL.cdf(z));
        }
    }

    /**
     * Calculates the Z-score approximation of the u-statistic
     * @param n - The number of entries in the stochastically smaller (dominant) set
     * @param m - The number of entries in the stochastically larger (dominated) set
     * @param u - the Mann-Whitney U value
     * @return the asymptotic z-approximation corresponding to the MWU p-value for n < m
     */
    @Requires({"m > 0","n > 0"})
    @Ensures({"! Double.isNaN(result)", "! Double.isInfinite(result)"})
    private static double getZApprox(int n, int m, double u, Map<Integer, Integer> tieStructure) {
        double mean = ( ((long)m)*n+1.0)/2;
        double var = (((long) n)*m*(n+m+1.0))/12;
        if (tieStructure != null) {
            double varAdjust = 0;
            for (Map.Entry<Integer, Integer> tie : tieStructure.entrySet()) {
                varAdjust += tie.getValue() * (Math.pow(tie.getKey(), 3) - tie.getKey());
            }
            var -= varAdjust / ((n + m) * (n + m - 1));
        }
        double z = ( u - mean )/Math.sqrt(var);
        return z;
    }

    /**
     * Uses a sum-of-uniform-0-1 random variable approximation to the U statistic in order to return an approximate
     * p-value. See Buckle, Kraft, van Eeden [1969] (approx) and Billingsly [1995] or Stephens, MA [1966, biometrika] (sum of uniform CDF)
     * @param n - The number of entries in the stochastically smaller (dominant) set
     * @param m - The number of entries in the stochastically larger (dominated) set
     * @param u - mann-whitney u value
     * @return p-value according to sum of uniform approx
     * todo -- this is currently not called due to not having a good characterization of where it is significantly more accurate than the
     * todo -- normal approxmation (e.g. enough to merit the runtime hit)
     */
    public static double calculatePUniformApproximation(int n, int m, double u) {
        double R = u + (n*(n+1))/2;
        double a = Math.sqrt(m*(n+m+1));
        double b = (n/2.0)*(1-Math.sqrt((n+m+1)/m));
        double z = b + R/a;
        if ( z < 0 ) { return 1.0; }
        else if ( z > n ) { return 0.0; }
        else {
            if ( z > ((double) n) /2 ) {
                return 1.0-1/(Arithmetic.factorial(n))*uniformSumHelper(z, (int) Math.floor(z), n, 0);
            } else {
                return 1/(Arithmetic.factorial(n))*uniformSumHelper(z, (int) Math.floor(z), n, 0);
            }
        }
    }

    /**
     * Helper function for the sum of n uniform random variables
     * @param z - value at which to compute the (un-normalized) cdf
     * @param m - a cutoff integer (defined by m <= z < m + 1)
     * @param n - the number of uniform random variables
     * @param k - holder variable for the recursion (alternatively, the index of the term in the sequence)
     * @return the (un-normalized) cdf for the sum of n random variables
     */
    private static double uniformSumHelper(double z, int m, int n, int k) {
        if ( k > m ) { return 0; }
        int coef = (k % 2 == 0) ? 1 : -1;
        return coef*Arithmetic.binomial(n,k)*Math.pow(z-k,n) + uniformSumHelper(z,m,n,k+1);
    }

    /**
     * Calculates the U-statistic associated with a two-sided test (e.g. the RV from which one set is drawn
     * stochastically dominates the RV from which the other set is drawn); two-sidedness is accounted for
     * later on simply by multiplying the p-value by 2.
     *
     * Recall: If X stochastically dominates Y, the test is for occurrences of Y before X, so the lower value of u is chosen
     * @param observed - the observed data
     * @return the minimum of the U counts (set1 dominates 2, set 2 dominates 1)
     */
    @Requires({"observed != null", "observed.size() > 0"})
    @Ensures({"result != null","result.first > 0"})
    public static Pair<Double,USet> calculateTwoSidedU(TreeMap<Number, Pair<Integer, Integer>> observed) {
        Pair<Double,Double> mwuResult = calculateMWU(observed);
        return mwuResult.second < mwuResult.first?
                new Pair<Double,USet>(mwuResult.first,USet.SET1) :
                new Pair<Double,USet>(mwuResult.second,USet.SET2);
    }

    /**
     * Calculates the U-statistic associated with the one-sided hypothesis that "dominator" stochastically dominates
     * the other U-set. Note that if S1 dominates S2, we want to count the occurrences of points in S2 coming before points in S1.
     * @param observed - the observed data points, tagged by each set
     * @param dominator - the set that is hypothesized to be stochastically dominating
     * @return the u-statistic associated with the hypothesis that dominator stochastically dominates the other set
     */
    @Requires({"observed != null","dominator != null","observed.size() > 0"})
    @Ensures({"result >= 0"})
    public static double calculateOneSidedU(TreeMap<Number, Pair<Integer, Integer>> observed, USet dominator) {
        Pair<Double,Double> mwuResult = calculateMWU(observed);
        return dominator == USet.SET1 ? mwuResult.first : mwuResult.second;
    }

    public static Map<Integer, Integer> getTieStructure(TreeMap<Number, Pair<Integer, Integer>> observed) {
        Map<Integer, Integer> tieStructure = new TreeMap<>();
        for (Pair<Integer, Integer> vals : observed.values()) {
            int useKey = vals.first + vals.second;

            Integer oldVal = tieStructure.get(useKey);
            if (oldVal == null) {
                oldVal = 0;
            }

            oldVal += 1;
            tieStructure.put(useKey, oldVal);
        }

        return tieStructure;
    }

    public static Pair<Double, Double> calculateMWU(TreeMap<Number, Pair<Integer, Integer>> observed) {
        int currRank = 0;

        int numDomObs1 = 0;
        double uDom1 = 0;
        int numDomObs2 = 0;
        double uDom2 = 0;

        for (Pair<Integer, Integer> numObsAtVal : observed.values()) {
            int totalObsAtVal = numObsAtVal.first + numObsAtVal.second;
            double valRank = currRank + (totalObsAtVal + 1) / 2.0;

            numDomObs1 += numObsAtVal.first;
            uDom1 += valRank * numObsAtVal.first;
            numDomObs2 += numObsAtVal.second;
            uDom2 += valRank * numObsAtVal.second;

            currRank += numObsAtVal.first + numObsAtVal.second;
        }

        uDom1 = uDom1 - numDomObs1 * (numDomObs1 + 1) / 2.0;
        uDom2 = uDom2 - numDomObs2 * (numDomObs2 + 1) / 2.0;

        return new Pair<Double, Double>(uDom1, uDom2);
    }


    /**
     * The Mann-Whitney U statistic follows a recursive equation (that enumerates the proportion of possible
     * binary strings of "n" zeros, and "m" ones, where a one precedes a zero "u" times). This accessor
     * calls into that recursive calculation.
     * @param n: number of set-one entries (hypothesis: set one is stochastically less than set two)
     * @param m: number of set-two entries
     * @param u: number of set-two entries that precede set-one entries (e.g. 0,1,0,1,0 -> 3 )
     * @param twoSided: whether the test is two sided or not. The recursive formula is symmetric, multiply by two for two-sidedness.
     * @param  mode: whether the mode is a point probability, or a cumulative distribution
     * @return the probability under the hypothesis that all sequences are equally likely of finding a set-two entry preceding a set-one entry "u" times.
     */
    @Requires({"m > 0","n > 0","u >= 0"})
    @Ensures({"result != null","! Double.isInfinite(result.getFirst())", "! Double.isInfinite(result.getSecond())"})
    public static Pair<Double,Double> calculatePRecursively(int n, int m, double u, boolean twoSided, ExactMode mode) {
        if ( m > 8 && n > 5 ) { throw new GATKException(String.format("Please use the appropriate (normal or sum of uniform) approximation. Values n: %d, m: %d",n,m)); }
        // N.B. We have found that simply using the floor along with the usual recursion (not adjusted for ties) produces entirely reasonable results in practice
        double p = mode == ExactMode.POINT ? cpr(n,m, (long) Math.floor(u)) : cumulativeCPR(n,m, (long) Math.floor(u));
        //p *= twoSided ? 2.0 : 1.0;
        double z;
        try {

            if ( mode == ExactMode.CUMULATIVE ) {
                z = APACHE_NORMAL.inverseCumulativeProbability(p);
            } else {
                double sd = Math.sqrt((1.0+1.0/(1+n+m))*(n*m)*(1.0+n+m)/12); // biased variance empirically better fit to distribution then asymptotic variance
                //System.out.printf("SD is %f and Max is %f and prob is %f%n",sd,1.0/Math.sqrt(sd*sd*2.0*Math.PI),p);
                if ( p > 1.0/Math.sqrt(sd*sd*2.0*Math.PI) ) { // possible for p-value to be outside the range of the normal. Happens at the mean, so z is 0.
                    z = 0.0;
                } else {
                    if ( u >= n*m/2 ) {
                        z = Math.sqrt(-2.0*(Math.log(sd)+Math.log(p)+LNSQRT2PI));
                    } else {
                        z = -Math.sqrt(-2.0*(Math.log(sd)+Math.log(p)+LNSQRT2PI));
                    }
                }
            }

        } catch (MathException me) {
            throw new GATKException("A math exception occurred in inverting the probability",me);
        }

        return new Pair<Double,Double>(z,(twoSided ? 2.0*p : p));
    }

    /**
     * Hook into CPR with sufficient warning (for testing purposes)
     * calls into that recursive calculation.
     * @param n: number of set-one entries (hypothesis: set one is stochastically less than set two)
     * @param m: number of set-two entries
     * @param u: number of set-two entries that precede set-one entries (e.g. 0,1,0,1,0 -> 3 )
     * @return same as cpr
     */
    protected static double calculatePRecursivelyDoNotCheckValuesEvenThoughItIsSlow(int n, int m, long u) {
        return cpr(n,m,u);
    }

    /**
     * For testing
     *
     * @param n: number of set-one entries (hypothesis: set one is stochastically less than set two)
     * @param m: number of set-two entries
     * @param u: number of set-two entries that precede set-one entries (e.g. 0,1,0,1,0 -> 3 )
     */
    protected static long countSequences(int n, int m, long u) {
        if ( u < 0 ) { return 0; }
        if ( m == 0 || n == 0 ) { return u == 0 ? 1 : 0; }

        return countSequences(n-1,m,u-m) + countSequences(n,m-1,u);
    }

    /**
     * : just a shorter name for calculatePRecursively. See Mann, Whitney, [1947]
     * @param n: number of set-1 entries
     * @param m: number of set-2 entries
     * @param u: number of times a set-2 entry as preceded a set-1 entry
     * @return recursive p-value
     */
    private static double cpr(int n, int m, long u) {
        if ( u < 0 ) {
            return 0.0;
        }
        if ( m == 0 || n == 0 ) {
            // there are entries in set 1 or set 2, so no set-2 entry can precede a set-1 entry; thus u must be zero.
            // note that this exists only for edification, as when we reach this point, the coefficient on this term is zero anyway
            return ( u == 0 ) ? 1.0 : 0.0;
        }


        return (((double)n)/(n+m))*cpr(n-1,m,u-m) + (((double)m)/(n+m))*cpr(n,m-1,u);
    }

    private static double cumulativeCPR(int n, int m, long u ) {
        // from above:
        // the null hypothesis is that {N} is stochastically less than {M}, so U has counted
        // occurrences of {M}s before {N}s. We would expect that this should be less than (n*m+1)/2 under
        // the null hypothesis, so we want to integrate from K=0 to K=U for cumulative cases. Always.
        double p = 0.0;
        // optimization using symmetry, use the least amount of sums possible
        double uSym = ( u <= n*m/2 ) ? u : ((long)n)*m-u;
        for ( long uu = 0; uu < uSym; uu++ ) {
            p += cpr(n,m,uu);
        }
        // correct by 1.0-p if the optimization above was used (e.g. 1-right tail = left tail)
        return (u <= n*m/2) ? p : 1.0-p;
    }

    /**
     * hook into the data tree, for testing purposes only
     * @return  observations
     */
    protected TreeMap<Number, Pair<Integer, Integer>> getObservations() {
        return observations;
    }

    /**
     * hook into the set sizes, for testing purposes only
     * @return size set 1, size set 2
     */
    protected Pair<Integer,Integer> getSetSizes() {
        return new Pair<Integer,Integer>(sizeSet1,sizeSet2);
    }

    /**
     * Validates that observations are in the correct format for a MWU test -- this is only called by the contracts API during testing
     * @param tree - the collection of labeled observations
     * @return true iff the tree set is valid (no INFs or NaNs, at least one data point in each set)
     */
    protected static boolean validateObservations(TreeMap<Number, Pair<Integer, Integer>> tree) {
        boolean seen1 = false;
        boolean seen2 = false;
        boolean seenInvalid = false;
        for ( Map.Entry<Number, Pair<Integer, Integer>> p : tree.entrySet()) {
            if ( ! seen1 && p.getValue().first > 0 ) {
                seen1 = true;
            }

            if ( ! seen2 && p.getValue().second > 0 ) {
                seen2 = true;
            }

            if ( Double.isNaN(p.getKey().doubleValue()) || Double.isInfinite(p.getKey().doubleValue())) {
                seenInvalid = true;
            }

        }

            return ! seenInvalid && seen1 && seen2;
    }

    public enum USet { SET1, SET2 }
    public enum ExactMode { POINT, CUMULATIVE }
}
