import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import Swal from 'sweetalert2';
import { CompanyFormComponent } from './company-form.component';

@Component({
  selector: 'app-company-list',
  templateUrl: './company-list.component.html',
  styleUrls: ['./company-list.component.css'],
  standalone: true,
  imports: [CommonModule, CompanyFormComponent]
})
export class CompanyListComponent implements OnInit {
  companies: any[] = [];
  selectedCompany: any = null; // To hold the company being edited
  formVisible: boolean = false; // To toggle the form visibility

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.fetchCompanies();
  }

  fetchCompanies() {
    this.http.get('http://localhost:3000/companies').subscribe(
      (data: any) => {
        this.companies = data;
      },
      error => {
        console.error('Error fetching companies:', error);
        Swal.fire('Error', 'There was an issue fetching the companies.', 'error');
      }
    );
  }

  onEdit(company: any) {
    this.selectedCompany = { ...company }; // Pre-fill the form with company data
    this.formVisible = true; // Show the form
  }

  onDelete(company: any) {
    Swal.fire({
      title: 'Are you sure?',
      text: `You are about to delete ${company.name}. This action cannot be undone.`,
      icon: 'warning',
      showCancelButton: true,
      confirmButtonText: 'Yes, delete it!',
      cancelButtonText: 'Cancel'
    }).then((result) => {
      if (result.isConfirmed) {
        this.http.delete(`http://localhost:3000/companies/${company.id}`).subscribe(
          () => {
            Swal.fire('Deleted!', `${company.name} has been deleted.`, 'success');
            this.companies = this.companies.filter(c => c.id !== company.id);
          },
          error => {
            Swal.fire('Error', 'There was an issue deleting the company.', 'error');
          }
        );
      }
    });
  }

  onFormSubmit(companyData: any) {
    if (this.selectedCompany) {
      // Editing an existing company
      this.http.put(`http://localhost:3000/companies/${this.selectedCompany.id}`, companyData).subscribe(
        (updatedCompany: any) => {
          Swal.fire('Updated!', `${updatedCompany.name} has been updated.`, 'success');
          const index = this.companies.findIndex(c => c.id === updatedCompany.id);
          if (index !== -1) {
            this.companies[index] = updatedCompany; // Update the list
          }
          this.resetForm();
        },
        error => {
          Swal.fire('Error', 'There was an issue updating the company.', 'error');
        }
      );
    } else {
      // Adding a new company
      this.http.post('http://localhost:3000/companies', companyData).subscribe(
        (newCompany: any) => {
          Swal.fire('Added!', `${newCompany.name} has been added.`, 'success');
          this.companies.push(newCompany); // Add the new company to the list
          this.resetForm();
        },
        error => {
          Swal.fire('Error', 'There was an issue adding the company.', 'error');
        }
      );
    }
  }

  resetForm() {
    this.selectedCompany = null;
    this.formVisible = false;
  }
}
