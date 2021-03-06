/*
* By downloading the PROGRAM you agree to the following terms of use:
* 
* BROAD INSTITUTE
* SOFTWARE LICENSE AGREEMENT
* FOR ACADEMIC NON-COMMERCIAL RESEARCH PURPOSES ONLY
* 
* This Agreement is made between the Broad Institute, Inc. with a principal address at 415 Main Street, Cambridge, MA 02142 ("BROAD") and the LICENSEE and is effective at the date the downloading is completed ("EFFECTIVE DATE").
* 
* WHEREAS, LICENSEE desires to license the PROGRAM, as defined hereinafter, and BROAD wishes to have this PROGRAM utilized in the public interest, subject only to the royalty-free, nonexclusive, nontransferable license rights of the United States Government pursuant to 48 CFR 52.227-14; and
* WHEREAS, LICENSEE desires to license the PROGRAM and BROAD desires to grant a license on the following terms and conditions.
* NOW, THEREFORE, in consideration of the promises and covenants made herein, the parties hereto agree as follows:
* 
* 1. DEFINITIONS
* 1.1 PROGRAM shall mean copyright in the object code and source code known as GATK3 and related documentation, if any, as they exist on the EFFECTIVE DATE and can be downloaded from http://www.broadinstitute.org/gatk on the EFFECTIVE DATE.
* 
* 2. LICENSE
* 2.1 Grant. Subject to the terms of this Agreement, BROAD hereby grants to LICENSEE, solely for academic non-commercial research purposes, a non-exclusive, non-transferable license to: (a) download, execute and display the PROGRAM and (b) create bug fixes and modify the PROGRAM. LICENSEE hereby automatically grants to BROAD a non-exclusive, royalty-free, irrevocable license to any LICENSEE bug fixes or modifications to the PROGRAM with unlimited rights to sublicense and/or distribute.  LICENSEE agrees to provide any such modifications and bug fixes to BROAD promptly upon their creation.
* The LICENSEE may apply the PROGRAM in a pipeline to data owned by users other than the LICENSEE and provide these users the results of the PROGRAM provided LICENSEE does so for academic non-commercial purposes only. For clarification purposes, academic sponsored research is not a commercial use under the terms of this Agreement.
* 2.2 No Sublicensing or Additional Rights. LICENSEE shall not sublicense or distribute the PROGRAM, in whole or in part, without prior written permission from BROAD. LICENSEE shall ensure that all of its users agree to the terms of this Agreement. LICENSEE further agrees that it shall not put the PROGRAM on a network, server, or other similar technology that may be accessed by anyone other than the LICENSEE and its employees and users who have agreed to the terms of this agreement.
* 2.3 License Limitations. Nothing in this Agreement shall be construed to confer any rights upon LICENSEE by implication, estoppel, or otherwise to any computer software, trademark, intellectual property, or patent rights of BROAD, or of any other entity, except as expressly granted herein. LICENSEE agrees that the PROGRAM, in whole or part, shall not be used for any commercial purpose, including without limitation, as the basis of a commercial software or hardware product or to provide services. LICENSEE further agrees that the PROGRAM shall not be copied or otherwise adapted in order to circumvent the need for obtaining a license for use of the PROGRAM.
* 
* 3. PHONE-HOME FEATURE
* LICENSEE expressly acknowledges that the PROGRAM contains an embedded automatic reporting system ("PHONE-HOME") which is enabled by default upon download. Unless LICENSEE requests disablement of PHONE-HOME, LICENSEE agrees that BROAD may collect limited information transmitted by PHONE-HOME regarding LICENSEE and its use of the PROGRAM.  Such information shall include LICENSEE'S user identification, version number of the PROGRAM and tools being run, mode of analysis employed, and any error reports generated during run-time.  Collection of such information is used by BROAD solely to monitor usage rates, fulfill reporting requirements to BROAD funding agencies, drive improvements to the PROGRAM, and facilitate adjustments to PROGRAM-related documentation.
* 
* 4. OWNERSHIP OF INTELLECTUAL PROPERTY
* LICENSEE acknowledges that title to the PROGRAM shall remain with BROAD. The PROGRAM is marked with the following BROAD copyright notice and notice of attribution to contributors. LICENSEE shall retain such notice on all copies. LICENSEE agrees to include appropriate attribution if any results obtained from use of the PROGRAM are included in any publication.
* Copyright 2012-2016 Broad Institute, Inc.
* Notice of attribution: The GATK3 program was made available through the generosity of Medical and Population Genetics program at the Broad Institute, Inc.
* LICENSEE shall not use any trademark or trade name of BROAD, or any variation, adaptation, or abbreviation, of such marks or trade names, or any names of officers, faculty, students, employees, or agents of BROAD except as states above for attribution purposes.
* 
* 5. INDEMNIFICATION
* LICENSEE shall indemnify, defend, and hold harmless BROAD, and their respective officers, faculty, students, employees, associated investigators and agents, and their respective successors, heirs and assigns, (Indemnitees), against any liability, damage, loss, or expense (including reasonable attorneys fees and expenses) incurred by or imposed upon any of the Indemnitees in connection with any claims, suits, actions, demands or judgments arising out of any theory of liability (including, without limitation, actions in the form of tort, warranty, or strict liability and regardless of whether such action has any factual basis) pursuant to any right or license granted under this Agreement.
* 
* 6. NO REPRESENTATIONS OR WARRANTIES
* THE PROGRAM IS DELIVERED AS IS. BROAD MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY KIND CONCERNING THE PROGRAM OR THE COPYRIGHT, EXPRESS OR IMPLIED, INCLUDING, WITHOUT LIMITATION, WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NONINFRINGEMENT, OR THE ABSENCE OF LATENT OR OTHER DEFECTS, WHETHER OR NOT DISCOVERABLE. BROAD EXTENDS NO WARRANTIES OF ANY KIND AS TO PROGRAM CONFORMITY WITH WHATEVER USER MANUALS OR OTHER LITERATURE MAY BE ISSUED FROM TIME TO TIME.
* IN NO EVENT SHALL BROAD OR ITS RESPECTIVE DIRECTORS, OFFICERS, EMPLOYEES, AFFILIATED INVESTIGATORS AND AFFILIATES BE LIABLE FOR INCIDENTAL OR CONSEQUENTIAL DAMAGES OF ANY KIND, INCLUDING, WITHOUT LIMITATION, ECONOMIC DAMAGES OR INJURY TO PROPERTY AND LOST PROFITS, REGARDLESS OF WHETHER BROAD SHALL BE ADVISED, SHALL HAVE OTHER REASON TO KNOW, OR IN FACT SHALL KNOW OF THE POSSIBILITY OF THE FOREGOING.
* 
* 7. ASSIGNMENT
* This Agreement is personal to LICENSEE and any rights or obligations assigned by LICENSEE without the prior written consent of BROAD shall be null and void.
* 
* 8. MISCELLANEOUS
* 8.1 Export Control. LICENSEE gives assurance that it will comply with all United States export control laws and regulations controlling the export of the PROGRAM, including, without limitation, all Export Administration Regulations of the United States Department of Commerce. Among other things, these laws and regulations prohibit, or require a license for, the export of certain types of software to specified countries.
* 8.2 Termination. LICENSEE shall have the right to terminate this Agreement for any reason upon prior written notice to BROAD. If LICENSEE breaches any provision hereunder, and fails to cure such breach within thirty (30) days, BROAD may terminate this Agreement immediately. Upon termination, LICENSEE shall provide BROAD with written assurance that the original and all copies of the PROGRAM have been destroyed, except that, upon prior written authorization from BROAD, LICENSEE may retain a copy for archive purposes.
* 8.3 Survival. The following provisions shall survive the expiration or termination of this Agreement: Articles 1, 3, 4, 5 and Sections 2.2, 2.3, 7.3, and 7.4.
* 8.4 Notice. Any notices under this Agreement shall be in writing, shall specifically refer to this Agreement, and shall be sent by hand, recognized national overnight courier, confirmed facsimile transmission, confirmed electronic mail, or registered or certified mail, postage prepaid, return receipt requested. All notices under this Agreement shall be deemed effective upon receipt.
* 8.5 Amendment and Waiver; Entire Agreement. This Agreement may be amended, supplemented, or otherwise modified only by means of a written instrument signed by all parties. Any waiver of any rights or failure to act in a specific instance shall relate only to such instance and shall not be construed as an agreement to waive any rights or fail to act in any other instance, whether or not similar. This Agreement constitutes the entire agreement among the parties with respect to its subject matter and supersedes prior agreements or understandings between the parties relating to its subject matter.
* 8.6 Binding Effect; Headings. This Agreement shall be binding upon and inure to the benefit of the parties and their respective permitted successors and assigns. All headings are for convenience only and shall not affect the meaning of any provision of this Agreement.
* 8.7 Governing Law. This Agreement shall be construed, governed, interpreted and applied in accordance with the internal laws of the Commonwealth of Massachusetts, U.S.A., without regard to conflict of laws principles.
*/

package org.broadinstitute.gatk.tools.walkers.haplotypecaller;

import com.google.common.annotations.VisibleForTesting;
import com.google.java.contract.Ensures;
import com.google.java.contract.Requires;
import htsjdk.samtools.util.StringUtil;
import htsjdk.variant.variantcontext.*;
import org.broadinstitute.gatk.engine.arguments.GenotypeCalculationArgumentCollection;
import org.broadinstitute.gatk.utils.*;
import org.broadinstitute.gatk.utils.contexts.ReferenceContext;
import org.broadinstitute.gatk.utils.genotyper.IndexedAlleleList;
import org.broadinstitute.gatk.utils.genotyper.SampleList;
import org.broadinstitute.gatk.utils.refdata.RefMetaDataTracker;
import org.broadinstitute.gatk.tools.walkers.genotyper.*;
import org.broadinstitute.gatk.tools.walkers.genotyper.afcalc.AFCalculatorProvider;
import org.broadinstitute.gatk.utils.collections.Pair;
import org.broadinstitute.gatk.utils.genotyper.ReadLikelihoods;
import org.broadinstitute.gatk.utils.haplotype.EventMap;
import org.broadinstitute.gatk.utils.haplotype.Haplotype;
import org.broadinstitute.gatk.utils.sam.GATKSAMRecord;
import org.broadinstitute.gatk.utils.variant.GATKVCFConstants;
import org.broadinstitute.gatk.utils.variant.GATKVariantContextUtils;

import java.util.*;

/**
 * {@link HaplotypeCaller}'s genotyping strategy implementation.
 */
public class HaplotypeCallerGenotypingEngine extends GenotypingEngine<AssemblyBasedCallerArgumentCollection> {

    protected static final int ALLELE_EXTENSION = 2;
    private static final String phase01 = "0|1";
    private static final String phase10 = "1|0";
    private static final int MAX_DROPPED_ALTERNATIVE_ALLELES_TO_LOG = 20;

    private MergeVariantsAcrossHaplotypes crossHaplotypeEventMerger;

    protected final boolean doPhysicalPhasing;

    private final GenotypingModel genotypingModel;

    private final PloidyModel ploidyModel;

    /**
     * {@inheritDoc}
     * @param configuration {@inheritDoc}
     * @param samples {@inheritDoc}
     * @param genomeLocParser {@inheritDoc}
     * @param doPhysicalPhasing whether to try physical phasing.
     */
    public HaplotypeCallerGenotypingEngine(final AssemblyBasedCallerArgumentCollection configuration, final SampleList samples, final GenomeLocParser genomeLocParser, final AFCalculatorProvider afCalculatorProvider, final boolean doPhysicalPhasing) {
        super(configuration,samples,genomeLocParser,afCalculatorProvider);
        if (genomeLocParser == null)
            throw new IllegalArgumentException("the genome location parser provided cannot be null");
        this.doPhysicalPhasing= doPhysicalPhasing;
        ploidyModel = new HomogeneousPloidyModel(samples,configuration.genotypeArgs.samplePloidy);
        genotypingModel = new InfiniteRandomMatingPopulationModel();
    }

    /**
     * Change the merge variant across haplotypes for this engine.
     *
     * @param crossHaplotypeEventMerger new merger, can be {@code null}.
     */
    public void setCrossHaplotypeEventMerger(final MergeVariantsAcrossHaplotypes crossHaplotypeEventMerger) {
        this.crossHaplotypeEventMerger = crossHaplotypeEventMerger;
    }

    @Override
    protected String callSourceString() {
        return "HC_call";
    }

    @Override
    protected boolean forceKeepAllele(final Allele allele) {
        return allele == GATKVCFConstants.NON_REF_SYMBOLIC_ALLELE ||
                configuration.genotypingOutputMode == GenotypingOutputMode.GENOTYPE_GIVEN_ALLELES ||
                configuration.emitReferenceConfidence != ReferenceConfidenceMode.NONE;
    }

    @Override
    protected boolean forceSiteEmission() {
        return configuration.outputMode == OutputMode.EMIT_ALL_SITES || configuration.genotypingOutputMode == GenotypingOutputMode.GENOTYPE_GIVEN_ALLELES;
    }

    /**
     * Carries the result of a call to #assignGenotypeLikelihoods
     */
    public static class CalledHaplotypes {
        private final List<VariantContext> calls;
        private final Set<Haplotype> calledHaplotypes;

        public CalledHaplotypes(final List<VariantContext> calls, final Set<Haplotype> calledHaplotypes) {
            if ( calls == null ) throw new IllegalArgumentException("calls cannot be null");
            if ( calledHaplotypes == null ) throw new IllegalArgumentException("calledHaplotypes cannot be null");
            if ( Utils.xor(calls.isEmpty(), calledHaplotypes.isEmpty()) )
                throw new IllegalArgumentException("Calls and calledHaplotypes should both be empty or both not but got calls=" + calls + " calledHaplotypes=" + calledHaplotypes);
            this.calls = calls;
            this.calledHaplotypes = calledHaplotypes;
        }

        /**
         * Get the list of calls made at this location
         * @return a non-null (but potentially empty) list of calls
         */
        public List<VariantContext> getCalls() {
            return calls;
        }

        /**
         * Get the set of haplotypes that we actually called (i.e., underlying one of the VCs in getCalls().
         * @return a non-null set of haplotypes
         */
        public Set<Haplotype> getCalledHaplotypes() {
            return calledHaplotypes;
        }
    }

    /**
     * Main entry point of class - given a particular set of haplotypes, samples and reference context, compute
     * genotype likelihoods and assemble into a list of variant contexts and genomic events ready for calling
     *
     * The list of samples we're working with is obtained from the readLikelihoods
     *
     * @param haplotypes                             Haplotypes to assign likelihoods to
     * @param readLikelihoods                       Map from reads->(haplotypes,likelihoods)
     * @param perSampleFilteredReadList              Map from sample to reads that were filtered after assembly and before calculating per-read likelihoods.
     * @param ref                                    Reference bytes at active region
     * @param refLoc                                 Corresponding active region genome location
     * @param activeRegionWindow                     Active window
     * @param genomeLocParser                        GenomeLocParser
     * @param activeAllelesToGenotype                Alleles to genotype
     * @param emitReferenceConfidence whether we should add a &lt;NON_REF&gt; alternative allele to the result variation contexts.
     *
     * @return                                       A CalledHaplotypes object containing a list of VC's with genotyped events and called haplotypes
     *
     */
    @Requires({"refLoc.containsP(activeRegionWindow)", "haplotypes.size() > 0"})
    @Ensures("result != null")
    // TODO - can this be refactored? this is hard to follow!
    CalledHaplotypes assignGenotypeLikelihoods( final List<Haplotype> haplotypes,
                                                       final ReadLikelihoods<Haplotype> readLikelihoods,
                                                       final Map<String, List<GATKSAMRecord>> perSampleFilteredReadList,
                                                       final byte[] ref,
                                                       final GenomeLoc refLoc,
                                                       final GenomeLoc activeRegionWindow,
                                                       final GenomeLocParser genomeLocParser,
                                                       final RefMetaDataTracker tracker,
                                                       final List<VariantContext> activeAllelesToGenotype,
                                                       final boolean emitReferenceConfidence) {
        // sanity check input arguments
        if (haplotypes == null || haplotypes.isEmpty()) throw new IllegalArgumentException("haplotypes input should be non-empty and non-null, got "+haplotypes);
        if (readLikelihoods == null || readLikelihoods.sampleCount() == 0) throw new IllegalArgumentException("readLikelihoods input should be non-empty and non-null, got "+readLikelihoods);
        if (ref == null || ref.length == 0 ) throw new IllegalArgumentException("ref bytes input should be non-empty and non-null, got " + Arrays.toString(ref));
        if (refLoc == null || refLoc.size() != ref.length) throw new IllegalArgumentException(" refLoc must be non-null and length must match ref bytes, got "+refLoc);
        if (activeRegionWindow == null ) throw new IllegalArgumentException("activeRegionWindow must be non-null");
        if (activeAllelesToGenotype == null ) throw new IllegalArgumentException("activeAllelesToGenotype must be non-null");
        if (genomeLocParser == null ) throw new IllegalArgumentException("genomeLocParser must be non-null");

        // update the haplotypes so we're ready to call, getting the ordered list of positions on the reference
        // that carry events among the haplotypes
        final TreeSet<Integer> startPosKeySet = decomposeHaplotypesIntoVariantContexts(haplotypes, readLikelihoods, ref, refLoc, activeAllelesToGenotype);

        // Walk along each position in the key set and create each event to be outputted
        final Set<Haplotype> calledHaplotypes = new HashSet<>();
        final List<VariantContext> returnCalls = new ArrayList<>();
        final int ploidy = configuration.genotypeArgs.samplePloidy;
        final List<Allele> noCallAlleles = GATKVariantContextUtils.noCallAlleles(ploidy);

        for( final int loc : startPosKeySet ) {
                // We're looping over all locations, even those outside the active region, looking for events that overlap the active region.

                final List<VariantContext> eventsAtThisLoc = new ArrayList<>();

                // Limit the event list to only those that overlap the active region
                for ( final VariantContext event : getVCsAtThisLocation(haplotypes, loc, activeAllelesToGenotype)) {
                    if ( genomeLocParser.createGenomeLoc(event).overlapsP(activeRegionWindow) ) {
                        eventsAtThisLoc.add(event);
                    }
                }

                if( eventsAtThisLoc.isEmpty() ) { continue; }

                // Create the event mapping object which maps the original haplotype events to the events present at just this locus
                final Map<Event, List<Haplotype>> eventMapper = createEventMapper(loc, eventsAtThisLoc, haplotypes);

                // Sanity check the priority list for mistakes
                final List<String> priorityList = makePriorityList(eventsAtThisLoc);

                // Merge the event to find a common reference representation

                VariantContext mergedVC = GATKVariantContextUtils.simpleMerge(eventsAtThisLoc, priorityList,
                        GATKVariantContextUtils.FilteredRecordMergeType.KEEP_IF_ANY_UNFILTERED,
                        GATKVariantContextUtils.GenotypeMergeType.PRIORITIZE, false, false, null, false, false);

                if( mergedVC == null )
                    continue;

                final GenotypeLikelihoodsCalculationModel.Model calculationModel = mergedVC.isSNP()
                        ? GenotypeLikelihoodsCalculationModel.Model.SNP : GenotypeLikelihoodsCalculationModel.Model.INDEL;

                final Map<VariantContext, Allele> mergeMap = new LinkedHashMap<>();
                mergeMap.put(null, mergedVC.getReference()); // the reference event (null) --> the reference allele
                for(int iii = 0; iii < eventsAtThisLoc.size(); iii++) {
                    mergeMap.put(eventsAtThisLoc.get(iii), mergedVC.getAlternateAllele(iii)); // BUGBUG: This is assuming that the order of alleles is the same as the priority list given to simpleMerge function
                }

                final Map<Allele, List<Haplotype>> alleleMapper = createAlleleMapper(mergeMap, eventMapper);

                if( configuration.DEBUG && logger != null ) {
                    if (logger != null) logger.info("Genotyping event at " + loc + " with alleles = " + mergedVC.getAlleles());
                }

                final ReadLikelihoods<Allele> readAlleleLikelihoods = readLikelihoods.marginalize(alleleMapper, genomeLocParser.createPaddedGenomeLoc(genomeLocParser.createGenomeLoc(mergedVC), ALLELE_EXTENSION));
                if (configuration.isSampleContaminationPresent())
                    readAlleleLikelihoods.contaminationDownsampling(configuration.getSampleContamination());

                final boolean someAllelesWereDropped = configuration.genotypeArgs.MAX_ALTERNATE_ALLELES < readAlleleLikelihoods.alleleCount() - 1;

                if (someAllelesWereDropped) {
                    reduceNumberOfAlternativeAllelesBasedOnLikelihoods(readAlleleLikelihoods, genomeLocParser.createGenomeLoc(mergedVC));
                }

                if (emitReferenceConfidence) {
                    mergedVC = addNonRefSymbolicAllele(mergedVC);
                    readAlleleLikelihoods.addNonReferenceAllele(GATKVCFConstants.NON_REF_SYMBOLIC_ALLELE);
                }

                final GenotypesContext genotypes = calculateGLsForThisEvent(readAlleleLikelihoods, noCallAlleles );
                final VariantContext call = calculateGenotypes(new VariantContextBuilder(mergedVC).alleles(readAlleleLikelihoods.alleles()).genotypes(genotypes).make(), calculationModel);
                if ( call != null ) {
                    final VariantContext annotatedCall = annotateCall(readLikelihoods, perSampleFilteredReadList, ref, refLoc, genomeLocParser, tracker, emitReferenceConfidence, calledHaplotypes, mergedVC, alleleMapper, readAlleleLikelihoods, someAllelesWereDropped, call);
                    returnCalls.add( annotatedCall );
            }
        }

        final List<VariantContext> phasedCalls = doPhysicalPhasing ? phaseCalls(returnCalls, calledHaplotypes) : returnCalls;
        return new CalledHaplotypes(phasedCalls, calledHaplotypes);
    }

    private VariantContext annotateCall(final ReadLikelihoods<Haplotype> readLikelihoods,
                                        final Map<String, List<GATKSAMRecord>> perSampleFilteredReadList,
                                        final byte[] ref, final GenomeLoc refLoc, final GenomeLocParser genomeLocParser,
                                        final RefMetaDataTracker tracker,
                                        final boolean emitReferenceConfidence,
                                        final Set<Haplotype> calledHaplotypes, final VariantContext mergedVC,
                                        final Map<Allele, List<Haplotype>> alleleMapper,
                                        final ReadLikelihoods<Allele> readAlleleLikelihoods,
                                        final boolean someAlternativeAllelesWereAlreadyDropped,
                                        final VariantContext call) {
        final int initialAlleleNumber = readAlleleLikelihoods.alleleCount();
        final ReadLikelihoods<Allele> readAlleleLikelihoodsForAnnotation = prepareReadAlleleLikelihoodsForAnnotation(readLikelihoods, perSampleFilteredReadList,
                genomeLocParser, emitReferenceConfidence, alleleMapper, readAlleleLikelihoods, call);

        ReferenceContext referenceContext = new ReferenceContext(genomeLocParser, genomeLocParser.createGenomeLoc(mergedVC), refLoc, ref);
        final boolean someAlternativeAllelesWereDropped = call.getAlleles().size() != initialAlleleNumber;
        VariantContext annotatedCall = annotationEngine.annotateContextForActiveRegion(referenceContext, tracker,readAlleleLikelihoodsForAnnotation, call, emitReferenceConfidence);
        if (someAlternativeAllelesWereDropped || someAlternativeAllelesWereAlreadyDropped)
           annotatedCall = GATKVariantContextUtils.reverseTrimAlleles(annotatedCall);

        // maintain the set of all called haplotypes
        for ( final Allele calledAllele : call.getAlleles() ) {
            final List<Haplotype> haplotypeList = alleleMapper.get(calledAllele);
            if (haplotypeList == null) continue;
            calledHaplotypes.addAll(haplotypeList);
        }

        return !emitReferenceConfidence ? clearUnconfidentGenotypeCalls(annotatedCall) : annotatedCall;
    }

    /**
     * Reduce the number alternative alleles in a read-likelihoods collection to the maximum-alt-allele user parameter value.
     * <p>
     *     We always keep the reference allele.
     *     As for the other alleles we keep the ones with the highest AF estimated as
     *     described in {@link #excessAlternativeAlleles(GenotypingLikelihoods, int)}
     * </p>
     * @param readAlleleLikelihoods the target read-likelihood collection.
     */
    private void reduceNumberOfAlternativeAllelesBasedOnLikelihoods(final ReadLikelihoods<Allele> readAlleleLikelihoods, final GenomeLoc location) {
        final GenotypingLikelihoods<Allele> genotypeLikelihoods = genotypingModel.calculateLikelihoods(readAlleleLikelihoods, new GenotypingData<>(ploidyModel,readAlleleLikelihoods));
        final Set<Allele> allelesToDrop = excessAlternativeAlleles(genotypeLikelihoods, configuration.genotypeArgs.MAX_ALTERNATE_ALLELES);
        final String allelesToDropString;
        if (allelesToDrop.size() <= MAX_DROPPED_ALTERNATIVE_ALLELES_TO_LOG) {
            allelesToDropString = StringUtil.join(", ", allelesToDrop);
        } else {
            final Iterator<Allele> it = allelesToDrop.iterator();
            final StringBuilder builder = new StringBuilder();
            for (int i = 0; i < MAX_DROPPED_ALTERNATIVE_ALLELES_TO_LOG; i++) {
                builder.append(it.next().toString()).append(", ");
            }
            allelesToDropString = builder.append(it.next().toString()).append(" and ").append(allelesToDrop.size() - 20).append(" more").toString();
        }
        logger.warn(String.format("location %s: too many alternative alleles found (%d) larger than the maximum requested with -%s (%d), the following will be dropped: %s.", location,
                readAlleleLikelihoods.alleleCount() - 1, GenotypeCalculationArgumentCollection.MAX_ALTERNATE_ALLELES_SHORT_NAME, configuration.genotypeArgs.MAX_ALTERNATE_ALLELES,
                allelesToDropString));
        readAlleleLikelihoods.dropAlleles(allelesToDrop);
    }

    /**
     * Returns the set of alleles that should be dropped in order to bring down the number
     * of alternative alleles to the maximum allowed.
     *
     * <p>
     *     The alleles that put forward for removal are those with the lowest estimated allele count.
     * </p>
     * <p>
     *     Allele counts are estimated herein as the weighted average count
     *     across samples and phased genotypes where the weight is the genotype likelihood-- we apply
     *     a uniform prior to all genotypes configurations.
     * </p>
     * <p>
     *     In case of a tie, unlikely for non trivial likelihoods, we keep the alleles with the lower index.
     * </p>
     *
     * @param genotypeLikelihoods target genotype likelihoods.
     * @param maxAlternativeAlleles maximum number of alternative alleles allowed.
     * @return never {@code null}.
     */
    private Set<Allele> excessAlternativeAlleles(final GenotypingLikelihoods<Allele> genotypeLikelihoods, final int maxAlternativeAlleles) {
        final int alleleCount = genotypeLikelihoods.alleleCount();
        final int excessAlternativeAlleleCount = Math.max(0, alleleCount - 1 - maxAlternativeAlleles);
        if (excessAlternativeAlleleCount <= 0) {
            return Collections.emptySet();
        }

        final double log10NumberOfAlleles = MathUtils.Log10Cache.get(alleleCount); // log10(Num of Alleles); e.g. log10(2) for diploids.
        final double[] log10EstimatedACs = new double[alleleCount]; // where we store the AC estimates.
        // Set allele counts to 0 (i.e. exp(-Inf)) at the start.
        Arrays.fill(log10EstimatedACs, Double.NEGATIVE_INFINITY);

        for (int i = 0; i < genotypeLikelihoods.sampleCount(); i++) {
            final GenotypeLikelihoodCalculator calculator = GenotypeLikelihoodCalculators.getInstance(genotypeLikelihoods.samplePloidy(i), alleleCount);
            final int numberOfUnphasedGenotypes = calculator.genotypeCount();
            // unphased genotype log10 likelihoods
            final double[] log10Likelihoods = genotypeLikelihoods.sampleLikelihoods(i).getAsVector();
            // total number of phased genotypes for all possible combinations of allele counts.
            final double log10NumberOfPhasedGenotypes = calculator.ploidy() * log10NumberOfAlleles;
            for (int j = 0; j < numberOfUnphasedGenotypes; j++) {
                final GenotypeAlleleCounts alleleCounts = calculator.genotypeAlleleCountsAt(j);
                // given the current unphased genotype, how many phased genotypes there are:
                final double log10NumberOfPhasedGenotypesForThisUnphasedGenotype = alleleCounts.log10CombinationCount();
                final double log10GenotypeLikelihood = log10Likelihoods[j];
                for (int k = 0; k < alleleCounts.distinctAlleleCount(); k++) {
                    final int alleleIndex = alleleCounts.alleleIndexAt(k);
                    final int alleleCallCount = alleleCounts.alleleCountAt(k);
                    final double log10AlleleCount = MathUtils.Log10Cache.get(alleleCallCount);
                    final double log10Weight = log10GenotypeLikelihood + log10NumberOfPhasedGenotypesForThisUnphasedGenotype
                            - log10NumberOfPhasedGenotypes;
                    // update the allele AC adding the contribution of this unphased genotype at this sample.
                    log10EstimatedACs[alleleIndex] = MathUtils.log10sumLog10(log10EstimatedACs[alleleIndex],
                            log10Weight + log10AlleleCount);
                }
            }
        }

        final PriorityQueue<Allele> lessFrequentFirst = new PriorityQueue<>(alleleCount, new Comparator<Allele>() {
            @Override
            public int compare(final Allele a1, final Allele a2) {
                final int index1 = genotypeLikelihoods.alleleIndex(a1);
                final int index2 = genotypeLikelihoods.alleleIndex(a2);
                final double freq1 = log10EstimatedACs[index1];
                final double freq2 = log10EstimatedACs[index2];
                if (freq1 != freq2) {
                    return Double.compare(freq1, freq2);
                } else {
                    return Integer.compare(index2, index1);
                }
            }
        });

        for (int i = 1; i < alleleCount; i++) {
            lessFrequentFirst.add(genotypeLikelihoods.alleleAt(i));
        }

        final Set<Allele> result = new HashSet<>(excessAlternativeAlleleCount);
        for (int i = 0; i < excessAlternativeAlleleCount; i++) {
            result.add(lessFrequentFirst.remove());
        }
        return result;
    }

    /**
     * Tries to phase the individual alleles based on pairwise comparisons to the other alleles based on all called haplotypes
     *
     * @param calls             the list of called alleles
     * @param calledHaplotypes  the set of haplotypes used for calling
     * @return a non-null list which represents the possibly phased version of the calls
     */
    protected List<VariantContext> phaseCalls(final List<VariantContext> calls, final Set<Haplotype> calledHaplotypes) {

        // construct a mapping from alternate allele to the set of haplotypes that contain that allele
        final Map<VariantContext, Set<Haplotype>> haplotypeMap = constructHaplotypeMapping(calls, calledHaplotypes);

        // construct a mapping from call to phase set ID
        final Map<VariantContext, Pair<Integer, String>> phaseSetMapping = new HashMap<>();
        final int uniqueCounterEndValue = constructPhaseSetMapping(calls, haplotypeMap, calledHaplotypes.size() - 1, phaseSetMapping);

        // we want to establish (potential) *groups* of phased variants, so we need to be smart when looking at pairwise phasing partners
        return constructPhaseGroups(calls, phaseSetMapping, uniqueCounterEndValue);
    }

    /**
     * Construct the mapping from alternate allele to the set of haplotypes that contain that allele
     *
     * @param originalCalls    the original unphased calls
     * @param calledHaplotypes  the set of haplotypes used for calling
     * @return non-null Map
     */
    @VisibleForTesting
    static Map<VariantContext, Set<Haplotype>> constructHaplotypeMapping(final List<VariantContext> originalCalls,
                                                                                   final Set<Haplotype> calledHaplotypes) {
        final Map<VariantContext, Set<Haplotype>> haplotypeMap = new HashMap<>(originalCalls.size());
        for ( final VariantContext call : originalCalls ) {
            // don't try to phase if there is not exactly 1 alternate allele
            if ( ! isBiallelic(call) ) {
                haplotypeMap.put(call, Collections.emptySet());
                continue;
            }

            // keep track of the haplotypes that contain this particular alternate allele
            final Set<Haplotype> hapsWithAllele = new HashSet<>();
            final Allele alt = call.getAlternateAllele(0);

            for ( final Haplotype h : calledHaplotypes ) {
                for ( final VariantContext event : h.getEventMap().getVariantContexts() ) {
                    if ( event.getStart() == call.getStart() && event.getAlternateAlleles().contains(alt) )
                        hapsWithAllele.add(h);
                }
            }
            haplotypeMap.put(call, hapsWithAllele);
        }

        return haplotypeMap;
    }


    /**
     * Construct the mapping from call (variant context) to phase set ID
     *
     * @param originalCalls    the original unphased calls
     * @param haplotypeMap     mapping from alternate allele to the set of haplotypes that contain that allele
     * @param totalAvailableHaplotypes the total number of possible haplotypes used in calling
     * @param phaseSetMapping  the map to populate in this method;
     *                         note that it is okay for this method NOT to populate the phaseSetMapping at all (e.g. in an impossible-to-phase situation)
     * @return the next incremental unique index
     */
    @VisibleForTesting
    static int constructPhaseSetMapping(final List<VariantContext> originalCalls,
                                        final Map<VariantContext, Set<Haplotype>> haplotypeMap,
                                        final int totalAvailableHaplotypes,
                                        final Map<VariantContext, Pair<Integer, String>> phaseSetMapping) {

        final int numCalls = originalCalls.size();
        int uniqueCounter = 0;

        // use the haplotype mapping to connect variants that are always/never present on the same haplotypes
        for ( int i = 0; i < numCalls - 1; i++ ) {
            final VariantContext call = originalCalls.get(i);
            final Set<Haplotype> haplotypesWithCall = haplotypeMap.get(call);
            if ( haplotypesWithCall.isEmpty() )
                continue;

            final boolean callIsOnAllHaps = haplotypesWithCall.size() == totalAvailableHaplotypes;

            for ( int j = i+1; j < numCalls; j++ ) {
                final VariantContext comp = originalCalls.get(j);
                final Set<Haplotype> haplotypesWithComp = haplotypeMap.get(comp);
                if ( haplotypesWithComp.isEmpty() )
                    continue;

                // if the variants are together on all haplotypes, record that fact.
                // another possibility is that one of the variants is on all possible haplotypes (i.e. it is homozygous).
                final boolean compIsOnAllHaps = haplotypesWithComp.size() == totalAvailableHaplotypes;
                if ( (haplotypesWithCall.size() == haplotypesWithComp.size() && haplotypesWithCall.containsAll(haplotypesWithComp)) || callIsOnAllHaps || compIsOnAllHaps ) {

                    // create a new group if these are the first entries
                    if ( ! phaseSetMapping.containsKey(call) ) {
                        // note that if the comp is already in the map then that is very bad because it means that there is
                        // another variant that is in phase with the comp but not with the call.  Since that's an un-phasable
                        // situation, we should abort if we encounter it.
                        if ( phaseSetMapping.containsKey(comp) ) {
                            phaseSetMapping.clear();
                            return 0;
                        }

                        // An important note: even for homozygous variants we are setting the phase as "0|1" here.
                        // We do this because we cannot possibly know for sure at this time that the genotype for this
                        // sample will actually be homozygous downstream: there are steps in the pipeline that are liable
                        // to change the genotypes.  Because we can't make those assumptions here, we have decided to output
                        // the phase as if the call is heterozygous and then "fix" it downstream as needed.
                        phaseSetMapping.put(call, new Pair<>(uniqueCounter, phase01));
                        phaseSetMapping.put(comp, new Pair<>(uniqueCounter, phase01));
                        uniqueCounter++;
                    }
                    // otherwise it's part of an existing group so use that group's unique ID
                    else if ( ! phaseSetMapping.containsKey(comp) ) {
                        final Pair<Integer, String> callPhase = phaseSetMapping.get(call);
                        phaseSetMapping.put(comp, new Pair<>(callPhase.first, callPhase.second));
                    }
                }
                // if the variants are apart on *all* haplotypes, record that fact
                else if ( haplotypesWithCall.size() + haplotypesWithComp.size() == totalAvailableHaplotypes ) {

                    final Set<Haplotype> intersection = new HashSet<>();
                    intersection.addAll(haplotypesWithCall);
                    intersection.retainAll(haplotypesWithComp);
                    if ( intersection.isEmpty() ) {
                        // create a new group if these are the first entries
                        if ( ! phaseSetMapping.containsKey(call) ) {
                            // note that if the comp is already in the map then that is very bad because it means that there is
                            // another variant that is in phase with the comp but not with the call.  Since that's an un-phasable
                            // situation, we should abort if we encounter it.
                            if ( phaseSetMapping.containsKey(comp) ) {
                                phaseSetMapping.clear();
                                return 0;
                            }

                            phaseSetMapping.put(call, new Pair<>(uniqueCounter, phase01));
                            phaseSetMapping.put(comp, new Pair<>(uniqueCounter, phase10));
                            uniqueCounter++;
                        }
                        // otherwise it's part of an existing group so use that group's unique ID
                        else if ( ! phaseSetMapping.containsKey(comp) ){
                            final Pair<Integer, String> callPhase = phaseSetMapping.get(call);
                            phaseSetMapping.put(comp, new Pair<>(callPhase.first, callPhase.second.equals(phase01) ? phase10 : phase01));
                        }
                    }
                }
            }
        }

        return uniqueCounter;
    }

    /**
     * Assemble the phase groups together and update the original calls accordingly
     *
     * @param originalCalls    the original unphased calls
     * @param phaseSetMapping  mapping from call (variant context) to phase group ID
     * @param indexTo          last index (exclusive) of phase group IDs
     * @return a non-null list which represents the possibly phased version of the calls
     */
    @VisibleForTesting
    static List<VariantContext> constructPhaseGroups(final List<VariantContext> originalCalls,
                                                     final Map<VariantContext, Pair<Integer, String>> phaseSetMapping,
                                                     final int indexTo) {
        final List<VariantContext> phasedCalls = new ArrayList<>(originalCalls);

        // if we managed to find any phased groups, update the VariantContexts
        for ( int count = 0; count < indexTo; count++ ) {
            // get all of the (indexes of the) calls that belong in this group (keeping them in the original order)
            final List<Integer> indexes = new ArrayList<>();
            for ( int index = 0; index < originalCalls.size(); index++ ) {
                final VariantContext call = originalCalls.get(index);
                if ( phaseSetMapping.containsKey(call) && phaseSetMapping.get(call).first == count )
                    indexes.add(index);
            }
            if ( indexes.size() < 2 )
                throw new IllegalStateException("Somehow we have a group of phased variants that has fewer than 2 members");

            // create a unique ID based on the leftmost one
            final String uniqueID = createUniqueID(originalCalls.get(indexes.get(0)));

            // update the VCs
            for ( final int index : indexes ) {
                final VariantContext originalCall = originalCalls.get(index);
                final VariantContext phasedCall = phaseVC(originalCall, uniqueID, phaseSetMapping.get(originalCall).second);
                phasedCalls.set(index, phasedCall);
            }
        }

        return phasedCalls;
    }

    /**
     * Is this variant bi-allelic?  This implementation is very much specific to this class so shouldn't be pulled out into a generalized place.
     *
     * @param vc the variant context
     * @return true if this variant context is bi-allelic, ignoring the NON-REF symbolic allele, false otherwise
     */
    private static boolean isBiallelic(final VariantContext vc) {
        return vc.isBiallelic() || (vc.getNAlleles() == 3 && vc.getAlternateAlleles().contains(GATKVCFConstants.NON_REF_SYMBOLIC_ALLELE));
    }

    /**
     * Create a unique identifier given the variant context
     *
     * @param vc   the variant context
     * @return non-null String
     */
    private static String createUniqueID(final VariantContext vc) {
        return String.format("%d_%s_%s", vc.getStart(), vc.getReference().getDisplayString(), vc.getAlternateAllele(0).getDisplayString());
        // return base + "_0," + base + "_1";
    }

    /**
     * Add physical phase information to the provided variant context
     *
     * @param vc   the variant context
     * @param ID   the ID to use
     * @param phaseGT the phase GT string to use
     * @return phased non-null variant context
     */
    private static VariantContext phaseVC(final VariantContext vc, final String ID, final String phaseGT) {
        final List<Genotype> phasedGenotypes = new ArrayList<>();
        for ( final Genotype g : vc.getGenotypes() )
            phasedGenotypes.add(new GenotypeBuilder(g).attribute(GATKVCFConstants.HAPLOTYPE_CALLER_PHASING_ID_KEY, ID).attribute(GATKVCFConstants.HAPLOTYPE_CALLER_PHASING_GT_KEY, phaseGT).make());
        return new VariantContextBuilder(vc).genotypes(phasedGenotypes).make();
    }

    private VariantContext addNonRefSymbolicAllele(final VariantContext mergedVC) {
        final VariantContextBuilder vcb = new VariantContextBuilder(mergedVC);
        final List<Allele> originalList = mergedVC.getAlleles();
        final List<Allele> alleleList = new ArrayList<>(originalList.size() + 1);
        alleleList.addAll(mergedVC.getAlleles());
        alleleList.add(GATKVCFConstants.NON_REF_SYMBOLIC_ALLELE);
        vcb.alleles(alleleList);
        return vcb.make();
    }

    // Builds the read-likelihoods collection to use for annotation considering user arguments and the collection
    // used for genotyping.
    protected ReadLikelihoods<Allele> prepareReadAlleleLikelihoodsForAnnotation(
            final ReadLikelihoods<Haplotype> readHaplotypeLikelihoods,
            final Map<String, List<GATKSAMRecord>> perSampleFilteredReadList,
            final GenomeLocParser genomeLocParser,
            final boolean emitReferenceConfidence,
            final Map<Allele, List<Haplotype>> alleleMapper,
            final ReadLikelihoods<Allele> readAlleleLikelihoodsForGenotyping,
            final VariantContext call) {

        final ReadLikelihoods<Allele> readAlleleLikelihoodsForAnnotations;
        final GenomeLoc loc = genomeLocParser.createGenomeLoc(call);

        // We can reuse for annotation the likelihood for genotyping as long as there is no contamination filtering
        // or the user want to use the contamination filtered set for annotations.
        // Otherwise (else part) we need to do it again.
        if (configuration.USE_FILTERED_READ_MAP_FOR_ANNOTATIONS || !configuration.isSampleContaminationPresent()) {
            readAlleleLikelihoodsForAnnotations = readAlleleLikelihoodsForGenotyping;
            readAlleleLikelihoodsForAnnotations.filterToOnlyOverlappingUnclippedReads(loc);
        } else {
            readAlleleLikelihoodsForAnnotations = readHaplotypeLikelihoods.marginalize(alleleMapper, loc);
            if (emitReferenceConfidence)
                readAlleleLikelihoodsForAnnotations.addNonReferenceAllele(
                        GATKVCFConstants.NON_REF_SYMBOLIC_ALLELE);
        }

        if (call.getAlleles().size() != readAlleleLikelihoodsForAnnotations.alleleCount()) {
            readAlleleLikelihoodsForAnnotations.updateNonRefAlleleLikelihoods(new IndexedAlleleList<>(new HashSet<>(call.getAlleles())));
        }

        // Skim the filtered map based on the location so that we do not add filtered read that are going to be removed
        // right after a few lines of code bellow.
        final Map<String, List<GATKSAMRecord>> overlappingFilteredReads = overlappingFilteredReads(perSampleFilteredReadList, loc);

        readAlleleLikelihoodsForAnnotations.addReads(overlappingFilteredReads,0);

        return readAlleleLikelihoodsForAnnotations;
    }


    private Map<String, List<GATKSAMRecord>> overlappingFilteredReads(final Map<String, List<GATKSAMRecord>> perSampleFilteredReadList, final GenomeLoc loc) {
        final Map<String,List<GATKSAMRecord>> overlappingFilteredReads = new HashMap<>(perSampleFilteredReadList.size());

        for (final Map.Entry<String,List<GATKSAMRecord>> sampleEntry : perSampleFilteredReadList.entrySet()) {
            final List<GATKSAMRecord> originalList = sampleEntry.getValue();
            final String sample = sampleEntry.getKey();
            if (originalList == null || originalList.size() == 0)
                continue;
            final List<GATKSAMRecord> newList = new ArrayList<>(originalList.size());
            for (final GATKSAMRecord read : originalList) {
                if (ReadLikelihoods.unclippedReadOverlapsRegion(read, loc))
                    newList.add(read);
            }
            if (newList.size() == 0)
                continue;
            overlappingFilteredReads.put(sample,newList);
        }
        return overlappingFilteredReads;
    }

    /**
     * Go through the haplotypes we assembled, and decompose them into their constituent variant contexts
     *
     * @param haplotypes the list of haplotypes we're working with
     * @param readLikelihoods map from samples -> the per read allele likelihoods
     * @param ref the reference bases (over the same interval as the haplotypes)
     * @param refLoc the span of the reference bases
     * @param activeAllelesToGenotype alleles we want to ensure are scheduled for genotyping (GGA mode)
     * @return never {@code null} but perhaps an empty list if there is no variants to report.
     */
    protected TreeSet<Integer> decomposeHaplotypesIntoVariantContexts(final List<Haplotype> haplotypes,
                                                                    final ReadLikelihoods readLikelihoods,
                                                                    final byte[] ref,
                                                                    final GenomeLoc refLoc,
                                                                    final List<VariantContext> activeAllelesToGenotype) {
        final boolean in_GGA_mode = !activeAllelesToGenotype.isEmpty();

        // Using the cigar from each called haplotype figure out what events need to be written out in a VCF file
        final TreeSet<Integer> startPosKeySet = EventMap.buildEventMapsForHaplotypes(haplotypes, ref, refLoc, configuration.DEBUG);

        if ( !in_GGA_mode ) {
            // run the event merger if we're not in GGA mode
            if (crossHaplotypeEventMerger == null)
                throw new IllegalStateException(" no variant merger was provided at set-up when needed in GGA mode");
            final boolean mergedAnything = crossHaplotypeEventMerger.merge(haplotypes, readLikelihoods, startPosKeySet, ref, refLoc);
            if ( mergedAnything )
                cleanUpSymbolicUnassembledEvents( haplotypes ); // the newly created merged events could be overlapping the unassembled events
        } else {
            startPosKeySet.clear();
            for( final VariantContext compVC : activeAllelesToGenotype ) {
                startPosKeySet.add( compVC.getStart() );
            }
        }

        return startPosKeySet;
    }

    /**
     * Get the priority list (just the list of sources for these variant context) used to merge overlapping events into common reference view
     * @param vcs a list of variant contexts
     * @return the list of the sources of vcs in the same order
     */
    protected List<String> makePriorityList(final List<VariantContext> vcs) {
        final List<String> priorityList = new LinkedList<>();
        for ( final VariantContext vc : vcs ) priorityList.add(vc.getSource());
        return priorityList;
    }

    protected List<VariantContext> getVCsAtThisLocation(final List<Haplotype> haplotypes,
                                                      final int loc,
                                                      final List<VariantContext> activeAllelesToGenotype) {
        // the overlapping events to merge into a common reference view
        final List<VariantContext> eventsAtThisLoc = new ArrayList<>();

        if( activeAllelesToGenotype.isEmpty() ) {
            for( final Haplotype h : haplotypes ) {
                final EventMap eventMap = h.getEventMap();
                final VariantContext vc = eventMap.get(loc);
                if( vc != null && !containsVCWithMatchingAlleles(eventsAtThisLoc, vc) ) {
                    eventsAtThisLoc.add(vc);
                }
            }
        } else { // we are in GGA mode!
            int compCount = 0;
            for( final VariantContext compVC : activeAllelesToGenotype ) {
                if( compVC.getStart() == loc ) {
                    int alleleCount = 0;
                    for( final Allele compAltAllele : compVC.getAlternateAlleles() ) {
                        List<Allele> alleleSet = new ArrayList<>(2);
                        alleleSet.add(compVC.getReference());
                        alleleSet.add(compAltAllele);
                        final String vcSourceName = "Comp" + compCount + "Allele" + alleleCount;
                        // check if this event is already in the list of events due to a repeat in the input alleles track
                        final VariantContext candidateEventToAdd = new VariantContextBuilder(compVC).alleles(alleleSet).source(vcSourceName).make();
                        boolean alreadyExists = false;
                        for( final VariantContext eventToTest : eventsAtThisLoc ) {
                            if( eventToTest.hasSameAllelesAs(candidateEventToAdd) ) {
                                alreadyExists = true;
                            }
                        }
                        if( !alreadyExists ) {
                            eventsAtThisLoc.add(candidateEventToAdd);
                        }
                        alleleCount++;
                    }
                }
                compCount++;
            }
        }

        return eventsAtThisLoc;
    }

    /**
     * For a particular event described in inputVC, form PL vector for each sample by looking into allele read map and filling likelihood matrix for each allele
     * @param readLikelihoods          Allele map describing mapping from reads to alleles and corresponding likelihoods
     * @return                       GenotypesContext object wrapping genotype objects with PLs
     */
    @Requires({"readLikelihoods!= null", "mergedVC != null"})
    @Ensures("result != null")
    private GenotypesContext  calculateGLsForThisEvent(final ReadLikelihoods<Allele> readLikelihoods, final List<Allele> noCallAlleles) {
        final GenotypingLikelihoods<Allele> likelihoods = genotypingModel.calculateLikelihoods(readLikelihoods, new GenotypingData<>(ploidyModel, readLikelihoods));
        final int sampleCount = samples.sampleCount();
        final GenotypesContext result = GenotypesContext.create(sampleCount);
        for (int s = 0; s < sampleCount; s++)
            result.add(new GenotypeBuilder(samples.sampleAt(s)).alleles(noCallAlleles).PL(likelihoods.sampleLikelihoods(s).getAsPLs()).make());
        return result;
    }

    /**
     * Removes symbolic events from list of haplotypes
     * @param haplotypes       Input/output list of haplotypes, before/after removal
     */
    // TODO - split into input haplotypes and output haplotypes as not to share I/O arguments
    @Requires("haplotypes != null")
    private static void cleanUpSymbolicUnassembledEvents(final List<Haplotype> haplotypes) {
        final List<Haplotype> haplotypesToRemove = new ArrayList<>();
        for( final Haplotype h : haplotypes ) {
            for( final VariantContext vc : h.getEventMap().getVariantContexts() ) {
                if( vc.isSymbolic() ) {
                    for( final Haplotype h2 : haplotypes ) {
                        for( final VariantContext vc2 : h2.getEventMap().getVariantContexts() ) {
                            if( vc.getStart() == vc2.getStart() && (vc2.isIndel() || vc2.isMNP()) ) { // unfortunately symbolic alleles can't currently be combined with non-point events
                                haplotypesToRemove.add(h);
                                break;
                            }
                        }
                    }
                }
            }
        }
        haplotypes.removeAll(haplotypesToRemove);
    }

    protected static Map<Allele, List<Haplotype>> createAlleleMapper( final Map<VariantContext, Allele> mergeMap, final Map<Event, List<Haplotype>> eventMap ) {
        final Map<Allele, List<Haplotype>> alleleMapper = new LinkedHashMap<>();
        for( final Map.Entry<VariantContext, Allele> entry : mergeMap.entrySet() ) {
            alleleMapper.put(entry.getValue(), eventMap.get(new Event(entry.getKey())));
        }
        return alleleMapper;
    }

    @Requires({"haplotypes.size() >= eventsAtThisLoc.size() + 1"})
    @Ensures({"result.size() == eventsAtThisLoc.size() + 1"})
    protected static Map<Event, List<Haplotype>> createEventMapper( final int loc, final List<VariantContext> eventsAtThisLoc, final List<Haplotype> haplotypes) {

        final Map<Event, List<Haplotype>> eventMapper = new LinkedHashMap<>(eventsAtThisLoc.size()+1);
        final Event refEvent = new Event(null);
        eventMapper.put(refEvent, new ArrayList<>());
        for( final VariantContext vc : eventsAtThisLoc ) {
            eventMapper.put(new Event(vc), new ArrayList<>());
        }

        for( final Haplotype h : haplotypes ) {
            if( h.getEventMap().get(loc) == null ) {
                eventMapper.get(refEvent).add(h);
            } else {
                for( final VariantContext vcAtThisLoc : eventsAtThisLoc ) {
                    if( h.getEventMap().get(loc).hasSameAllelesAs(vcAtThisLoc) ) {
                        eventMapper.get(new Event(vcAtThisLoc)).add(h);
                        break;
                    }
                }
            }
        }

        return eventMapper;
    }

    @Deprecated
    @VisibleForTesting
    static Map<Integer,VariantContext> generateVCsFromAlignment(final Haplotype haplotype, final byte[] ref, final GenomeLoc refLoc, final String sourceNameToAdd) {
        return new EventMap(haplotype, ref, refLoc, sourceNameToAdd);
    }

    private static boolean containsVCWithMatchingAlleles( final List<VariantContext> list, final VariantContext vcToTest ) {
        for( final VariantContext vc : list ) {
            if( vc.hasSameAllelesAs(vcToTest) ) {
                return true;
            }
        }
        return false;
    }

    protected static class Event {
        public VariantContext vc;

        @VisibleForTesting
        Event( final VariantContext vc ) {
            this.vc = vc;
        }

        @Override
        public boolean equals( final Object obj ) {
            return obj instanceof Event && ((((Event) obj).vc == null && vc == null) || (((Event) obj).vc != null && vc != null && ((Event) obj).vc.hasSameAllelesAs(vc))) ;
        }

        @Override
        public int hashCode() {
            return (vc == null ? -1 : vc.getAlleles().hashCode());
        }
    }

    /**
     * Returns the ploidy-model used by this genotyping engine.
     *
     * @return never {@code null}.
     */
    PloidyModel getPloidyModel() {
        return ploidyModel;
    }

    /**
     * Returns the genotyping-model used by this genotyping engine.
     *
     * @return never {@code null}.
     */
    GenotypingModel getGenotypingModel() {
        return genotypingModel;
    }

    /**
     * Cleans up genotype-level annotations that need to be updated
     * (similar to {@link org.broadinstitute.gatk.tools.walkers.variantutils.GenotypeGVCFs#cleanupGenotypeAnnotations},
     * but here we only check for unconfident calls (i.e., those with GQ = 0), which are set to no-call).
     */
    private VariantContext clearUnconfidentGenotypeCalls(final VariantContext VC) {
        final GenotypesContext oldGTs = VC.getGenotypes();
        final List<Genotype> clearedGTs = new ArrayList<>(oldGTs.size());
        for ( final Genotype oldGT : oldGTs ) {
            // set GT to no-call when GQ is 0
            if (oldGT.hasGQ() && oldGT.getGQ() == 0) {
                final int ploidy = oldGT.getPloidy();
                final List<Allele> noCallAlleles = GATKVariantContextUtils.noCallAlleles(ploidy);
                final Genotype noCallGT = new GenotypeBuilder().alleles(noCallAlleles).make();
                clearedGTs.add(noCallGT);
            } else {
                clearedGTs.add(oldGT);
            }
        }
        return new VariantContextBuilder(VC).genotypes(clearedGTs).make();
    }
}