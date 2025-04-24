// src/app/components/question-bank/question-bank.component.ts
import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-question-bank',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './question-bank.component.html',
  styleUrls: ['./question-bank.component.css']
})

export class QuestionBankComponent {
  questionText = '';
  optionA = '';
  optionB = '';
  optionC = '';
  optionD = '';
  correctOption = 'A';
  difficulty = 'easy';
  imageFile: File | null = null;

  constructor(private api: ApiService) {}

  onFileChange(event: any) {
    if (event.target.files && event.target.files.length) {
      this.imageFile = event.target.files[0];
    }
  }

  addQuestion() {
    const formData = new FormData();
    formData.append('question_text', this.questionText);
    formData.append('option_a', this.optionA);
    formData.append('option_b', this.optionB);
    formData.append('option_c', this.optionC);
    formData.append('option_d', this.optionD);
    formData.append('correct_option', this.correctOption);
    formData.append('difficulty', this.difficulty);
    if (this.imageFile) {
      formData.append('image', this.imageFile);
    }
    this.api.addQuestion(formData).subscribe(response => {
      alert('Question added successfully!');
      // Reset form fields
      this.questionText = '';
      this.optionA = '';
      this.optionB = '';
      this.optionC = '';
      this.optionD = '';
        this.correctOption = 'A';
      this.difficulty = 'easy';
      this.imageFile = null;
    });
  }
}
