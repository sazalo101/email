export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { email_text } = req.body;

  if (!email_text || !email_text.trim()) {
    return res.status(400).json({ error: 'Email text required' });
  }

  const result = detectSpam(email_text);
  res.status(200).json(result);
}

function detectSpam(text) {
  const spamKeywords = {
    urgency: ['urgent', 'immediate', 'act now', 'limited time', 'expires', 'deadline', 'within', 'hours', 'minutes', 'before', 'face arrest', 'avoid'],
    money: ['money', 'cash', 'prize', 'winner', 'lottery', 'million', 'thousand', 'owe', 'refund', 'payment', 'taxes', 'fees'],
    free: ['free', 'no cost', 'complimentary', 'gratis'],
    callToAction: ['click', 'call', 'download', 'visit', 'claim', 'order', 'verify', 'update', 'contact', 'restore', 'log in', 'confirm'],
    personalInfo: ['bank', 'account', 'password', 'ssn', 'credit', 'social security', 'identity', 'billing', 'benefits', 'activity', 'login'],
    threats: ['arrest', 'warrant', 'seizure', 'suspended', 'restricted', 'compromise', 'compromised', 'lien', 'court', 'summons', 'penalty', 'detected', 'breach'],
    authority: ['irs', 'federal', 'government', 'medicare', 'social security', 'bank of america', 'paypal', 'amazon', 'wells fargo', 'microsoft', 'apple', 'chase', 'citibank'],
    phishingVerbs: ['suspicious', 'detected', 'restricted', 'expires', 'required', 'immediately', 'verify', 'confirm', 'update', 'restore']
  };

  const spamDomains = ['secure-', 'verify-', '-secure', '-verification', '-update', '-portal', '-center'];
  const lowerText = text.toLowerCase();
  let spamScore = 0;
  const flaggedKeywords = [];
  const customFlags = [];

  // Check keywords
  Object.entries(spamKeywords).forEach(([category, keywords]) => {
    let categoryMatches = 0;
    keywords.forEach(keyword => {
      if (lowerText.includes(keyword.toLowerCase())) {
        categoryMatches++;
        if (!flaggedKeywords.includes(keyword)) {
          flaggedKeywords.push(keyword);
        }
      }
    });

    const weights = { urgency: 15, money: 12, threats: 20, authority: 18, callToAction: 8, personalInfo: 10, phishingVerbs: 15, free: 5 };
    spamScore += categoryMatches * (weights[category] || 5);

    if (categoryMatches > 0) {
      switch(category) {
        case 'authority': customFlags.push('authority_impersonation'); break;
        case 'threats': customFlags.push('threat_language_detected'); break;
        case 'phishingVerbs': customFlags.push('phishing_language_detected'); break;
        case 'urgency': customFlags.push('urgency_tactics'); break;
      }
    }
  });

  // Check domains
  if (spamDomains.some(domain => lowerText.includes(domain))) {
    spamScore += 25;
    customFlags.push('suspicious_domains_detected');
  }

  // Check formatting
  const upperCaseRatio = (text.match(/[A-Z]/g) || []).length / text.length;
  if (upperCaseRatio > 0.3) {
    spamScore += 15;
    customFlags.push('excessive_capitalization');
  }

  if ((text.match(/!/g) || []).length > 2) {
    spamScore += 10;
    customFlags.push('excessive_punctuation');
  }

  // Calculate probability
  let probability = Math.min(100, spamScore / 2);

  // Boost sophisticated phishing
  if (customFlags.includes('authority_impersonation') && 
      customFlags.includes('threat_language_detected') && 
      (customFlags.includes('suspicious_domains_detected') || customFlags.includes('phishing_language_detected'))) {
    probability = Math.max(probability, 85);
  }

  const isSpam = probability > 50;
  let confidence = probability > 80 || probability < 20 ? 'High' : probability > 60 || probability < 40 ? 'Medium' : 'Low';

  return {
    spam_probability: Math.round(probability * 10) / 10,
    is_spam: isSpam,
    classification: isSpam ? 'SPAM' : 'NOT SPAM',
    confidence: confidence,
    flagged_keywords: flaggedKeywords.slice(0, 10),
    custom_flags: [...new Set(customFlags)]
  };
}