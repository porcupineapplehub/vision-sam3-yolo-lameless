/**
 * Language Context
 * Provides English/French language switching
 */
import { createContext, useContext, useEffect, useState, ReactNode } from 'react'

export type Language = 'en' | 'fr'

interface LanguageContextType {
  language: Language
  setLanguage: (lang: Language) => void
  t: (key: string) => string
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined)

const LANGUAGE_STORAGE_KEY = 'lameness-language'

// Translation dictionary
const translations: Record<Language, Record<string, string>> = {
  en: {
    // Navigation
    'nav.dashboard': 'Dashboard',
    'nav.cowRegistry': 'Cow Registry',
    'nav.pairwise': 'Pairwise',
    'nav.triplet': 'Triplet',
    'nav.hierarchy': 'Hierarchy',
    'nav.similarity': 'Similarity',
    'nav.learn': 'Learn',
    
    // Pairwise Comparison
    'pairwise.title': 'Pairwise Comparison',
    'pairwise.subtitle': 'Compare videos using a 7-point scale to build a lameness hierarchy',
    'pairwise.leftCow': 'Left Cow',
    'pairwise.rightCow': 'Right Cow',
    'pairwise.play': 'Play',
    'pairwise.pause': 'Pause',
    'pairwise.restart': 'Restart',
    'pairwise.selectOption': 'Select the option that best describes the lameness difference',
    'pairwise.leftMoreLame': 'Left more lame',
    'pairwise.rightMoreLame': 'Right more lame',
    'pairwise.submit': 'Submit & Next Pair (Enter)',
    'pairwise.submitting': 'Submitting...',
    'pairwise.showRanking': 'Show Ranking',
    'pairwise.share': 'Share',
    'pairwise.retakeTutorial': 'Retake Tutorial',
    'pairwise.skipTutorial': 'Skip Tutorial',
    'pairwise.loading': 'Loading video pair...',
    'pairwise.allComplete': 'All Comparisons Complete!',
    'pairwise.allCompleteMsg': "You've completed all pairwise comparisons. Great work!",
    'pairwise.goToDashboard': 'Go to Dashboard',
    
    // Tutorial
    'tutorial.title': 'Pairwise Comparison Tutorial',
    'tutorial.step': 'Step',
    'tutorial.of': 'of',
    'tutorial.correct': 'Correct!',
    'tutorial.notQuite': 'Not quite right',
    'tutorial.checkAnswer': 'Check Answer',
    'tutorial.nextExample': 'Next Example',
    'tutorial.startReal': 'Start Real Comparisons',
    'tutorial.score': 'Score:',
    
    // Triplet
    'triplet.title': 'Triplet Comparison',
    'triplet.subtitle': 'Help train our AI by identifying which cow is more similar to the reference',
    'triplet.reference': 'Reference Cow',
    'triplet.whichMoreSimilar': 'Which cow is more similar to the reference?',
    'triplet.comparisonA': 'Cow A',
    'triplet.comparisonB': 'Cow B',
    'triplet.confidence': 'Confidence',
    'triplet.high': 'High',
    'triplet.medium': 'Medium',
    'triplet.low': 'Low',
    'triplet.allComplete': 'All Triplet Tasks Complete!',
    'triplet.allCompleteMsg': "You've completed all triplet comparisons. Great work!",
    'triplet.goToPairwise': 'Go to Pairwise Comparison',
    
    // Performance
    'performance.title': 'Performance',
    'performance.subtitle': 'Your rating performance compared to the consensus',
    'performance.totalRatings': 'Total Ratings',
    'performance.accuracy': 'Accuracy',
    'performance.meanDeviation': 'Mean Deviation',
    'performance.consistency': 'Consistency',
    'performance.distributionTitle': 'Your Performance Distribution',
    'performance.distributionDesc': 'Your ratings compared to the majority consensus. You are within',
    'performance.distributionDescEnd': 'standard deviations from the mean.',
    'performance.you': 'YOU',
    'performance.otherRaters': 'Other raters',
    'performance.yourPosition': 'Your position',
    'performance.sigmaNote': 'σ (sigma) represents standard deviation from the mean (μ)',
    'performance.insightsTitle': 'Performance Insights',
    'performance.goodConsistency': 'Good Consistency',
    'performance.consistencyDesc': 'Your ratings are closely aligned with expert consensus (within',
    'performance.aboveAverage': 'Above Average Performance',
    'performance.aboveAverageDesc': "You're performing better than",
    'performance.aboveAverageDescEnd': 'of raters',
    'performance.keepRating': 'Keep Rating',
    'performance.keepRatingDesc': 'Complete more comparisons to improve your consistency score',
    
    // Common
    'common.progress': 'Progress:',
    'common.pairs': 'pairs',
    'common.tasks': 'tasks',
    'common.loading': 'Loading...',
    'common.settings': 'Settings',
    'common.logout': 'Logout',
    'common.search': 'Search...',
    'common.quickLinks': 'Quick Links',
  },
  fr: {
    // Navigation
    'nav.dashboard': 'Tableau de bord',
    'nav.cowRegistry': 'Registre des vaches',
    'nav.pairwise': 'Comparaison par paires',
    'nav.triplet': 'Triplet',
    'nav.hierarchy': 'Hiérarchie',
    'nav.similarity': 'Similarité',
    'nav.learn': 'Apprendre',
    
    // Pairwise Comparison
    'pairwise.title': 'Comparaison par paires',
    'pairwise.subtitle': 'Comparez les vidéos à l\'aide d\'une échelle de 7 points pour construire une hiérarchie de boiterie',
    'pairwise.leftCow': 'Vache gauche',
    'pairwise.rightCow': 'Vache droite',
    'pairwise.play': 'Lecture',
    'pairwise.pause': 'Pause',
    'pairwise.restart': 'Redémarrer',
    'pairwise.selectOption': 'Sélectionnez l\'option qui décrit le mieux la différence de boiterie',
    'pairwise.leftMoreLame': 'Gauche plus boiteuse',
    'pairwise.rightMoreLame': 'Droite plus boiteuse',
    'pairwise.submit': 'Soumettre et paire suivante (Entrée)',
    'pairwise.submitting': 'Envoi...',
    'pairwise.showRanking': 'Afficher le classement',
    'pairwise.share': 'Partager',
    'pairwise.retakeTutorial': 'Refaire le tutoriel',
    'pairwise.skipTutorial': 'Passer le tutoriel',
    'pairwise.loading': 'Chargement de la paire de vidéos...',
    'pairwise.allComplete': 'Toutes les comparaisons terminées!',
    'pairwise.allCompleteMsg': 'Vous avez terminé toutes les comparaisons par paires. Excellent travail!',
    'pairwise.goToDashboard': 'Aller au tableau de bord',
    
    // Tutorial
    'tutorial.title': 'Tutoriel de comparaison par paires',
    'tutorial.step': 'Étape',
    'tutorial.of': 'sur',
    'tutorial.correct': 'Correct!',
    'tutorial.notQuite': 'Pas tout à fait',
    'tutorial.checkAnswer': 'Vérifier la réponse',
    'tutorial.nextExample': 'Exemple suivant',
    'tutorial.startReal': 'Commencer les vraies comparaisons',
    'tutorial.score': 'Score:',
    
    // Triplet
    'triplet.title': 'Comparaison en triplet',
    'triplet.subtitle': 'Aidez à former notre IA en identifiant quelle vache est la plus similaire à la référence',
    'triplet.reference': 'Vache de référence',
    'triplet.whichMoreSimilar': 'Quelle vache est la plus similaire à la référence?',
    'triplet.comparisonA': 'Vache A',
    'triplet.comparisonB': 'Vache B',
    'triplet.confidence': 'Confiance',
    'triplet.high': 'Haute',
    'triplet.medium': 'Moyenne',
    'triplet.low': 'Faible',
    'triplet.allComplete': 'Toutes les tâches en triplet terminées!',
    'triplet.allCompleteMsg': 'Vous avez terminé toutes les comparaisons en triplet. Excellent travail!',
    'triplet.goToPairwise': 'Aller à la comparaison par paires',
    
    // Performance
    'performance.title': 'Performance',
    'performance.subtitle': 'Votre performance de notation par rapport au consensus',
    'performance.totalRatings': 'Évaluations totales',
    'performance.accuracy': 'Précision',
    'performance.meanDeviation': 'Déviation moyenne',
    'performance.consistency': 'Cohérence',
    'performance.distributionTitle': 'Distribution de votre performance',
    'performance.distributionDesc': 'Vos évaluations comparées au consensus majoritaire. Vous êtes à',
    'performance.distributionDescEnd': 'écarts-types de la moyenne.',
    'performance.you': 'VOUS',
    'performance.otherRaters': 'Autres évaluateurs',
    'performance.yourPosition': 'Votre position',
    'performance.sigmaNote': 'σ (sigma) représente l\'écart-type par rapport à la moyenne (μ)',
    'performance.insightsTitle': 'Aperçu des performances',
    'performance.goodConsistency': 'Bonne cohérence',
    'performance.consistencyDesc': 'Vos évaluations sont étroitement alignées sur le consensus d\'experts (à',
    'performance.aboveAverage': 'Performance supérieure à la moyenne',
    'performance.aboveAverageDesc': 'Vous performez mieux que',
    'performance.aboveAverageDescEnd': 'des évaluateurs',
    'performance.keepRating': 'Continuez à évaluer',
    'performance.keepRatingDesc': 'Complétez plus de comparaisons pour améliorer votre score de cohérence',
    
    // Common
    'common.progress': 'Progrès:',
    'common.pairs': 'paires',
    'common.tasks': 'tâches',
    'common.loading': 'Chargement...',
    'common.settings': 'Paramètres',
    'common.logout': 'Déconnexion',
    'common.search': 'Rechercher...',
    'common.quickLinks': 'Liens rapides',
  },
}

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<Language>(() => {
    if (typeof window === 'undefined') return 'en'
    return (localStorage.getItem(LANGUAGE_STORAGE_KEY) as Language) || 'en'
  })

  const setLanguage = (newLanguage: Language) => {
    setLanguageState(newLanguage)
    localStorage.setItem(LANGUAGE_STORAGE_KEY, newLanguage)
  }

  const t = (key: string): string => {
    return translations[language][key] || key
  }

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  )
}

export function useLanguage() {
  const context = useContext(LanguageContext)
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider')
  }
  return context
}
