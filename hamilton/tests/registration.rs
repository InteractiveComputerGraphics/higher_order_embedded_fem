use hamilton::{register_factory, GenericFactory, RegistrationStatus};

#[test]
fn register() {
    // Important: registration is global, so we must run this test in a separate binary,
    // which we do when we make it a separate integration test
    let make_factory = || Box::new(GenericFactory::<i32>::default());
    let make_factory2 = || Box::new(GenericFactory::<i64>::default());

    assert_eq!(register_factory(make_factory()).unwrap(), RegistrationStatus::Inserted);
    assert_eq!(register_factory(make_factory()).unwrap(), RegistrationStatus::Replaced);
    assert_eq!(register_factory(make_factory()).unwrap(), RegistrationStatus::Replaced);

    assert_eq!(register_factory(make_factory2()).unwrap(), RegistrationStatus::Inserted);
    assert_eq!(register_factory(make_factory2()).unwrap(), RegistrationStatus::Replaced);

    assert_eq!(register_factory(make_factory()).unwrap(), RegistrationStatus::Replaced);
}
