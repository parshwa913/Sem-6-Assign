// src/app/components/exam/exam.component.ts
import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { ApiService } from '../../services/api.service';


interface Exam {
  id: number;
  title: string;
  description: string;
  question_ids: number[];
  questions?: any[];
  duration_minutes: number;
}

@Component({
  selector: 'app-exam',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './exam.component.html',
  styleUrls: ['./exam.component.css']
})

export class ExamComponent implements OnInit {
  exam!: Exam;

  answers: { [questionId: number]: string } = {};
  examId: number | null = null;

  constructor(private route: ActivatedRoute, private api: ApiService) {}

  ngOnInit(): void {
    this.route.params.subscribe(params => {
      this.examId = +params['id'];
      this.loadExam();
    });
  }

  loadExam() {
    if (this.examId) {
      this.api.getExamById(this.examId).subscribe((data: any) => {
        this.exam = data;
        if (this.exam && this.exam.question_ids) {
          this.api.getQuestionsByIds(this.exam.question_ids).subscribe((questions: any[]) => {
            if (this.exam) {
              this.exam.questions = questions;
            }
          });
        }
      });
    }
  }

  recordAnswer(questionId: number, answer: string) {
    this.answers[questionId] = answer;
  }

  submitExam() {
    if (this.exam) {
      const submission = {
        exam_id: this.exam.id,
        answers: this.answers
      };
      this.api.submitExam(submission).subscribe((response: any) => {
        alert('Exam submitted! Your score is: ' + response.score);
      });
    }
  }
}
