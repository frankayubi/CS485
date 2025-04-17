import os
import json
import re
from collections import Counter

class QuestionTracker:
    def __init__(self, file_path='user_questions.json', max_questions=8):
        self.file_path = file_path
        self.max_questions = max_questions
        self.questions = self._load_questions()
        
    def _load_questions(self):
        """Load the questions from the JSON file."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except:
                return {"questions": []}
        else:
            return {"questions": []}
    
    def _save_questions(self):
        """Save the questions to the JSON file."""
        with open(self.file_path, 'w') as f:
            json.dump(self.questions, f)
    
    def add_question(self, question):
        """Add a question to the tracker."""
        # Clean the question
        question = question.strip()
        
        # Skip very short questions or commands
        if len(question) < 10 or not question.endswith('?'):
            return
        
        # Add to the list
        self.questions["questions"].append(question)
        self._save_questions()
    
    def get_most_common(self):
        """Get the most commonly asked questions."""
        # Simple similarity check - group questions that are nearly identical
        cleaned_questions = []
        
        for q in self.questions["questions"]:
            # Only include questions (ends with ?)
            if q.endswith('?'):
                # Clean up question for comparison
                cleaned = re.sub(r'[^\w\s\?]', '', q.lower())
                cleaned_questions.append(cleaned)
        
        # Count occurrences
        counter = Counter(cleaned_questions)
        
        # Get the most common questions
        common_questions = [q for q, _ in counter.most_common(self.max_questions)]
        
        # If we don't have enough questions yet, add some defaults
        if len(common_questions) < self.max_questions:
            defaults = [
                "What is diabetes?",
                "What are symptoms of acne?",
                "What is blood pressure?", 
                "What are symptoms of fever?",
                "What is asthma?",
                "What are symptoms of allergy?",
                "What is depression?",
                "What are symptoms of anxiety?"
            ]
            
            # Add defaults that aren't already in our list
            for q in defaults:
                if len(common_questions) >= self.max_questions:
                    break
                if q not in common_questions:
                    common_questions.append(q)
        
        return common_questions[:self.max_questions]