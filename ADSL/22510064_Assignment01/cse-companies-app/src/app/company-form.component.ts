import { Component, Input, OnChanges, Output, EventEmitter, SimpleChanges } from '@angular/core';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import Swal from 'sweetalert2';

@Component({
  selector: 'app-company-form',
  templateUrl: './company-form.component.html',
  styleUrls: ['./company-form.component.css'],
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule]
})
export class CompanyFormComponent implements OnChanges {
  @Input() company: any; // Pre-filled data for editing
  @Output() companyUpdated = new EventEmitter<any>(); // Event emitter to notify the parent component
  companyForm: FormGroup;
  isEditMode = false;

  constructor(private fb: FormBuilder, private http: HttpClient) {
    // Initialize the form with default values
    this.companyForm = this.fb.group({
      id: [null],
      name: ['', Validators.required],
      location: ['', Validators.required],
      employee_count: [0, [Validators.required, Validators.min(1)]],
      founded_year: ['', Validators.required],
      industry: ['', Validators.required]
    });
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['company'] && this.company) {
      this.isEditMode = true; // Switch to edit mode
      this.companyForm.patchValue(this.company); // Pre-fill form with company data
    } else {
      this.isEditMode = false; // New company mode
      this.companyForm.reset(); // Clear the form
    }
  }

  onSubmit() {
    if (this.companyForm.valid) {
      const endpoint = this.isEditMode
        ? `http://localhost:3000/companies/${this.companyForm.value.id}`
        : 'http://localhost:3000/companies';
      const method = this.isEditMode ? 'put' : 'post';

      this.http[method](endpoint, this.companyForm.value).subscribe(
        response => {
          Swal.fire(
            this.isEditMode ? 'Updated!' : 'Saved!',
            `Company ${this.isEditMode ? 'updated' : 'saved'} successfully.`,
            'success'
          );
          this.companyUpdated.emit(response); // Notify the parent about the update
          this.resetForm();
        },
        error => {
          Swal.fire('Error', 'There was an issue saving/updating the company.', 'error');
        }
      );
    }
  }

  resetForm() {
    this.companyForm.reset();
    this.isEditMode = false;
  }
}
